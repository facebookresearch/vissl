---
name: "Unexpected behaviors"
about: Run into unexpected behaviors when using VISSL
title: Please read & provide the following

---

If you do not know the root cause of the problem, and wish someone to help you, please
post according to this template:

## Instructions To Reproduce the Issue:

Check https://stackoverflow.com/help/minimal-reproducible-example for how to ask good questions.
Simplify the steps to reproduce the issue using suggestions from the above link, and provide them below:

1. full code you wrote or full changes you made (`git diff`)
```
<put code or diff here>
```
2. what exact command you run:
3. __full logs__ you observed:
```
<put logs here>
```

## Expected behavior:

If there are no obvious error in "what you observed" provided above,
please tell us the expected behavior.

If you expect the model to converge / work better, note that we do not give suggestions
on how to train a new model.
Only in one of the two conditions, we will help with it:
(1) You're unable to reproduce the results in vissl model zoo.
(2) It indicates a vissl bug.

## Environment:

Provide your environment information using the following command:
```
wget -nc -q https://github.com/facebookresearch/vissl/raw/main/vissl/utils/collect_env.py && python collect_env.py
```

If your issue looks like an installation issue / environment issue,
please first try to solve it with the instructions in
https://github.com/facebookresearch/vissl/tree/main/docs
