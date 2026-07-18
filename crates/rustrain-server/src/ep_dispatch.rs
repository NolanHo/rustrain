use std::fmt;

use tokio::sync::{mpsc, oneshot};

const DEFAULT_QUEUE_CAPACITY: usize = 32;
const HARD_MAX_QUEUE_CAPACITY: usize = 4096;

type DispatchJob = Box<dyn FnOnce() + Send + 'static>;

#[derive(Clone)]
pub(crate) struct EpDispatchScheduler {
    sender: mpsc::Sender<DispatchJob>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EpDispatchScheduleError {
    QueueFull,
    QueueClosed,
    WorkerFailed,
}

impl fmt::Display for EpDispatchScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull => formatter.write_str("EP dispatch queue is full"),
            Self::QueueClosed => formatter.write_str("EP dispatch queue is closed"),
            Self::WorkerFailed => formatter.write_str("EP dispatch worker failed"),
        }
    }
}

impl EpDispatchScheduler {
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "EP dispatch queue capacity must be positive");
        let (sender, mut receiver) = mpsc::channel::<DispatchJob>(capacity);
        tokio::spawn(async move {
            while let Some(job) = receiver.recv().await {
                if let Err(error) = tokio::task::spawn_blocking(job).await {
                    tracing::error!(%error, "EP dispatch job panicked");
                }
            }
        });
        Self { sender }
    }

    pub(crate) async fn run<T, F>(&self, operation: F) -> Result<T, EpDispatchScheduleError>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let receiver = self.submit(operation)?;
        receiver
            .await
            .map_err(|_| EpDispatchScheduleError::WorkerFailed)
    }

    fn submit<T, F>(&self, operation: F) -> Result<oneshot::Receiver<T>, EpDispatchScheduleError>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let (response, receiver) = oneshot::channel();
        let job = Box::new(move || {
            let _ = response.send(operation());
        });
        self.sender.try_send(job).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => EpDispatchScheduleError::QueueFull,
            mpsc::error::TrySendError::Closed(_) => EpDispatchScheduleError::QueueClosed,
        })?;
        Ok(receiver)
    }
}

pub(crate) fn configured_queue_capacity() -> usize {
    std::env::var("RUSTRAIN_EP_DISPATCH_QUEUE_CAPACITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|capacity| *capacity > 0)
        .unwrap_or(DEFAULT_QUEUE_CAPACITY)
        .min(HARD_MAX_QUEUE_CAPACITY)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};

    use tokio::sync::oneshot;

    use super::{EpDispatchScheduleError, EpDispatchScheduler};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn accepted_jobs_are_fifo_and_single_inflight() {
        let scheduler = EpDispatchScheduler::new(4);
        let order = Arc::new(Mutex::new(Vec::new()));
        let inflight = Arc::new(AtomicUsize::new(0));
        let max_inflight = Arc::new(AtomicUsize::new(0));

        let mut receivers = Vec::new();
        for index in 0..3 {
            let order = Arc::clone(&order);
            let inflight = Arc::clone(&inflight);
            let max_inflight = Arc::clone(&max_inflight);
            receivers.push(
                scheduler
                    .submit(move || {
                        let current = inflight.fetch_add(1, Ordering::SeqCst) + 1;
                        max_inflight.fetch_max(current, Ordering::SeqCst);
                        order.lock().unwrap().push(index);
                        inflight.fetch_sub(1, Ordering::SeqCst);
                        index
                    })
                    .unwrap(),
            );
        }

        for (index, receiver) in receivers.into_iter().enumerate() {
            assert_eq!(receiver.await.unwrap(), index);
        }
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2]);
        assert_eq!(max_inflight.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn full_queue_rejects_without_waiting() {
        let scheduler = EpDispatchScheduler::new(1);
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (started, started_rx) = oneshot::channel();
        let job_gate = Arc::clone(&gate);
        let first = scheduler
            .submit(move || {
                let _ = started.send(());
                let (lock, wake) = &*job_gate;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = wake.wait(released).unwrap();
                }
            })
            .unwrap();
        started_rx.await.unwrap();

        let second = scheduler.submit(|| 2usize).unwrap();
        assert_eq!(
            scheduler.submit(|| 3usize).unwrap_err(),
            EpDispatchScheduleError::QueueFull
        );

        let (lock, wake) = &*gate;
        *lock.lock().unwrap() = true;
        wake.notify_one();
        first.await.unwrap();
        assert_eq!(second.await.unwrap(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn panicking_job_does_not_stop_the_consumer() {
        let scheduler = EpDispatchScheduler::new(2);
        assert_eq!(
            scheduler.run(|| panic!("injected dispatch panic")).await,
            Err(EpDispatchScheduleError::WorkerFailed)
        );
        assert_eq!(scheduler.run(|| 7usize).await.unwrap(), 7);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn blocking_job_does_not_block_the_async_runtime() {
        let scheduler = EpDispatchScheduler::new(1);
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let job_gate = Arc::clone(&gate);
        let (started, started_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            scheduler
                .run(move || {
                    let _ = started.send(());
                    let (lock, wake) = &*job_gate;
                    let mut released = lock.lock().unwrap();
                    while !*released {
                        released = wake.wait(released).unwrap();
                    }
                })
                .await
        });
        started_rx.await.unwrap();

        tokio::time::timeout(std::time::Duration::from_millis(100), async {
            tokio::task::yield_now().await;
        })
        .await
        .expect("blocking dispatch must not occupy the async runtime thread");

        let (lock, wake) = &*gate;
        *lock.lock().unwrap() = true;
        wake.notify_one();
        task.await.unwrap().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_response_does_not_cancel_an_accepted_job() {
        let scheduler = EpDispatchScheduler::new(1);
        let (completed, completed_rx) = oneshot::channel();
        let response = scheduler
            .submit(move || {
                let _ = completed.send(());
                9usize
            })
            .unwrap();
        drop(response);

        tokio::time::timeout(std::time::Duration::from_secs(1), completed_rx)
            .await
            .expect("accepted dispatch must execute after its caller disconnects")
            .unwrap();
    }
}
