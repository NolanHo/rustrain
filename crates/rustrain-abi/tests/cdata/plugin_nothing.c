/*
 * A perfectly loadable shared object that exports no `rustrain_plugin_v1`.
 * The loader must report a missing entry symbol (contract C-1) rather than
 * silently treating the library as an empty plugin.
 */
int rustrain_fixture_not_the_entry(void) {
    return 0;
}
