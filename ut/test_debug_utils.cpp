#include "debug.h"


// build using: ./build_ut.sh -DLOG_MIN_LEVEL=LOG_LEVEL_ERROR -DLOG_LEVEL=LOG_LEVEL_DEBUG -DLOG_TAG=LOG_TAG_COPPER_NODE
int main() {
    CLOG(LOG_LEVEL_DEBUG, LOG_TAG_ANY, "Hello World! This is a DEBUG.");
    CLOG(LOG_LEVEL_LOG, LOG_TAG_ANY, "Hello World! This is a LOG.");
    CLOG(LOG_LEVEL_WARNING, LOG_TAG_ANY, "Hello World! This is a WARNING.");
    CLOG(LOG_LEVEL_ERROR, LOG_TAG_ANY, "Hello World! This is a ERROR.");
    
    CLOG(LOG_LEVEL_DEBUG, LOG_TAG_DEFAULT, "Hello World! This is a DEBUG With DEFAULT TAG.");


    AssertError(true, LOG_TAG_ANY, "This is a true assert error");
    AssertError(false, LOG_TAG_ANY, "This is a false assert error %d %s", 50, "hi");
    AssertFatal(true, LOG_TAG_ANY, "This is a true assert fatal");
    // AssertFatal(false, LOG_TAG_ANY, "This is a true assert fatal");
    CLOG(LOG_LEVEL_PANIC, LOG_TAG_ANY, "Hello World! This is a PANIC.");
}