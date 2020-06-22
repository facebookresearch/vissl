## Unit testing

To run the unittests in fbcode,

```bash
cd ~/fbsource/fbcode
buck test //deeplearning/projects/ssl_framework/tests:
```

To run an individual test:

```bash
cd ~/fbsource/fbcode
buck run //deeplearning/projects/ssl_framework/tests:{test_filename}
```
