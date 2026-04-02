GAMMAS := 0.9 0.95 0.99
TAUS := 0.005 0.01 0.05

LAMBDA := 0.9 0.95 1.00
EPSILON := 0.1 0.2 0.3

# Required on macOS to find the Homebrew GLFW dylib used by MuJoCo
GLFW_ENV := DYLD_LIBRARY_PATH=/opt/homebrew/lib:$$DYLD_LIBRARY_PATH

# Generate the list of targets
# TARGETS := $(foreach g,$(GAMMAS),$(foreach t,$(TAUS),run_$(g)_$(t)))
TARGETS := $(foreach g,$(LAMBDA),$(foreach t,$(EPSILON),run_$(g)_$(t)))

.PHONY: all $(TARGETS)

all: $(TARGETS)

$(TARGETS):
	$(eval G=$(word 2,$(subst _, ,$@)))
	$(eval T=$(word 3,$(subst _, ,$@)))
	$(eval INDEX=$(shell python3 -c "print('$(TARGETS)'.split().index('$@'))"))
	@echo "Scheduling PPO (Lambda=$(G), Epsilon=$(T)) with $(INDEX)s delay..."
	@sleep $(INDEX) && $(GLFW_ENV) uv run ./agents/ppo.py --env_id='Hopper-v4' --gae_lambda=$(G) --clip_coef=$(T)
