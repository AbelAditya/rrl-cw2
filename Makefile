GAMMAS := 0.9 0.95 0.99
TAUS := 0.005 0.01 0.05

# Generate the list of targets
TARGETS := $(foreach g,$(GAMMAS),$(foreach t,$(TAUS),run_$(g)_$(t)))

.PHONY: all $(TARGETS)

all: $(TARGETS)

$(TARGETS):
	$(eval G=$(word 2,$(subst _, ,$@)))
	$(eval T=$(word 3,$(subst _, ,$@)))
	$(eval INDEX=$(shell python3 -c "print('$(TARGETS)'.split().index('$@'))"))
	@echo "Scheduling SAC (Gamma=$(G), Tau=$(T)) with $(INDEX)s delay..."
	@sleep $(INDEX) && uv run ./agents/sac.py --env_id='Hopper-v4' --gamma=$(G) --tau=$(T)
