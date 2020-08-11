# 0.0.5 - 2020-08-11

- Expose `max_iter_harmony` as a new top-level argument, in addition to the
  previously exposed `max_iter_kmeans`. This more closely resembles the
  original interface in the harmony R package. Thanks to @pinin4fjords
  for pull request #8

# 0.0.4 - 2020-03-02

- Fix a bug in the LISI code that sometimes causes computation to break. Thanks
  to @tariqdaouda for reporting it in issue #1

- Fix a bug that prevents controlling the number of iterations. Thanks to
  @liboxun for reporting it in issue #3

- Fix a bug causing slightly different results than expected. Thanks to
  @bli25broad for pull request #2

- Add support for multiple categorical batch variables.

# 0.0.3 - 2019-12-26

- Speed up the Harmony algorithm. It should now be as fast as the R package.

# 0.0.2 - 2019-12-20

- Add Local Inverse Simpson Index (LISI) functions from the lisi R package.
  <https://github.com/immunogenomics/LISI>

# 0.0.1 - 2019-12-19

- Initial release. Code ported directly from the harmony R package.
  <https://github.com/immunogenomics/harmony>
