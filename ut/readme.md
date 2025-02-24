# Building Tests:
To build the tests, first run the build_ut.sh script and pass it the necessary arguments.
To rebuild the tests without cleaning the build and out directories, use fbuild_ut.sh instead.

## Values
* LOG_LEVEL:                Provides a hierarchy level for logs
    * LOG_LEVEL_ZERO     0 -> Includes all logs. It is not a log level itself.
    * LOG_LEVEL_PANIC    1 -> Prints the logs and panics.
    * LOG_LEVEL_ERROR    2
    * LOG_LEVEL_WARNING  3
    * LOG_LEVEL_LOG      4
    * LOG_LEVEL_DEBUG    5

* LOG_TAG:                  Is a bitmap used to ignore different subsets of logs
    * LOG_TAG_DEFAULT             0b00001
    * LOG_TAG_NOT_IMPLEMENTED     0b00010
    * LOG_TAG_TEST                0b00100
    * LOG_TAG_BASIC               0b01000
    * LOG_TAG_COPPER_NODE         0b10000
    * LOG_TAG_ANY                 0b11111

## Test Arguments:
    -DLOG_MIN_LEVEL=[LOG_LEVEL]             If the log level is equal or below MIN_LEVEL, LOG_TAG is ignored and log is printed. 
                                            Default value is LOG_LEVEL_ZERO (i.e. LOG_LEVEL does not affect TAG filtering).
    -DLOG_LEVEL=[LOG_LEVEL]                 All logs with levels above LOG_LEVEL will be ignored.
                                            Should be greater than or equal to LOG_MIN_LEVEL.
                                            Default value is LOG_LEVEL_LOG (i.e. Only the DEBUG logs are not printed).
    -DLOG_TAG=[LOG_TAG[ | LOG_TAG]...]      Only the logs which have their respective bit set to 1, will be printed.
                                            Default value is LOG_TAG_ANY (i.e. No logs are ignored based on TAG).
    -DOUT=[stdout|stderr]                   Sets the output stream for the logs. Default value is stdout.
    -DASSERT_ERROR_PANIC                    If used, all AssertErrors will act similar to AssertFatals 
                                            (i.e. They will crash the program if enabled). It is not defined by default.
## Locked Arguments:
    -DBUILD=[DEBUG|RELEASE]                 Is always set to DEBUG for Test builds.
    -DENABLE_TEST_LOGGING                   If defined, will enable all of the logs. It is always defined for Test builds.
    -DENABLE_ASSERTS                        If defined, will enable all of the asserts. It is always defined for Test builds.

# Running Tests:
To run the tests, you can use the run_ut.sh script and pass it the name of the test that you want to run.
For example running './run_ut.sh test_debug_utils' will run the tests associated with test_debug_utils.cpp.

## Notes:
* If ENABLE_TEST_LOGGING is not defined but ENABLE_ASSERTS is, then all asserts will be changed to simple asserts
    and all tags will be ignored. Moreover, In this case, if ASSERT_ERROR_PANIC is defined, AssertError will 
    behave similar to AssertFatal. Otherwise, they are also ignored.

## TODO
Currently, run_ut.sh only accepts one file per run. Therefore, we can add the option to:
* Pass multiple files for running the tests.
* Pass an argument(e.g. all) to tell the script to run all of the tests.
