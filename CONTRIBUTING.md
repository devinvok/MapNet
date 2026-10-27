# Contributing

Issues and pull requests are welcome.

- Run `pytest` before submitting. The suite needs no model, GPU or network, but a full run is slow (a few minutes).
- Small, focused changes are easiest to review.
- Benchmark claims should come with a run of `python -m realtest.run` (see `realtest/README.md`) so results can be compared.
- Good first areas: Windows and encoding fixes in the test harness, new question sets for other document collections, and ideas for better file descriptions.
