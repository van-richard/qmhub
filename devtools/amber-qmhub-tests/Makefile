PYTHON ?= python3
THIS_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
QMHUB_AMBER_TEST_WORK ?= $(THIS_DIR)/work

.PHONY: help test.serial test.parallel clean

help:
	@echo "External Amber/QMHub smoke tests"
	@echo ""
	@echo "Targets:"
	@echo "  make test.serial    Run patched serial sander/sqm/QMHub checks"
	@echo "  make test.parallel  Run patched sander.MPI checks"
	@echo "  make clean          Remove the default local work directory"
	@echo ""
	@echo "Useful variables:"
	@echo "  AMBERHOME=/path/to/amber/install"
	@echo "  QMHUB_AMBER_PRMTOP=/path/to/system.prmtop"
	@echo "  QMHUB_AMBER_RST7=/path/to/system.rst7"
	@echo "  QMHUB_AMBER_QMMASK=':1'"
	@echo "  QMHUB_AMBER_QMCHARGE=0"
	@echo "  QMHUB_AMBER_SPIN=1"
	@echo "  MPI_LAUNCH='mpirun -np 2'"
	@echo "  QMHUB_RUN_QCHEM=0|1"
	@echo "  QMHUB_KEEP_WORK=1"
	@echo "  QMHUB_AMBER_TEST_WORK=/path/to/workdir"

test.serial:
	QMHUB_AMBER_TEST_WORK="$(QMHUB_AMBER_TEST_WORK)" $(PYTHON) "$(THIS_DIR)/run_amber_qmhub_tests.py" --mode serial

test.parallel:
	QMHUB_AMBER_TEST_WORK="$(QMHUB_AMBER_TEST_WORK)" $(PYTHON) "$(THIS_DIR)/run_amber_qmhub_tests.py" --mode parallel

clean:
	rm -rf "$(QMHUB_AMBER_TEST_WORK)"
