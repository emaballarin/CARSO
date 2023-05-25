#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------

.PHONY: clean
clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	rm -R -f ./.mypy_cache

.PHONY: format
format:
	find . -type f -name '*.py' -exec reorder-python-imports --py310-plus "{}" \;
	black "$(realpath .)"
