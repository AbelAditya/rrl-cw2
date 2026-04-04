GAMMAS := 0.9 0.95 0.99
TAUS := 0.005 0.01 0.05

LAMBDA := 0.9 0.95 1.00
EPSILON := 0.1 0.2 0.3

# Required on macOS to find the Homebrew GLFW dylib used by MuJoCo
GLFW_ENV := DYLD_LIBRARY_PATH=/opt/homebrew/lib:$$DYLD_LIBRARY_PATH

# Generate the list of sweep targets
TARGETS := $(foreach g,$(LAMBDA),$(foreach t,$(EPSILON),run_$(g)_$(t)))

.PHONY: sweep $(TARGETS)

sweep: $(TARGETS)

$(TARGETS):
	$(eval G=$(word 2,$(subst _, ,$@)))
	$(eval T=$(word 3,$(subst _, ,$@)))
	$(eval INDEX=$(shell python3 -c "print('$(TARGETS)'.split().index('$@'))"))
	@echo "Scheduling PPO (Lambda=$(G), Epsilon=$(T)) with $(INDEX)s delay..."
	@sleep $(INDEX) && $(GLFW_ENV) uv run ./agents/ppo.py --env_id='Hopper-v4' --gae_lambda=$(G) --clip_coef=$(T)

# ─── Final runs ────────────────────────────────────────────────────────────────

PPY        := uv run python
SCRIPT     := agents/ppo.py
SCRIPT2    := agents/sac.py
ENV        := Ant-v4
CLIP_COEF  := 0.3
GAE_LAMBDA := 0.95
TAU        := 0.05
GAMMA      := 0.99

FINAL_TARGETS := \
	run-seed-1-ppo \
	run-seed-2-ppo \
	run-seed-3-ppo \
	run-seed-1-sac \
	run-seed-2-sac \
	run-seed-3-sac

.PHONY: all $(FINAL_TARGETS)

all: $(FINAL_TARGETS)

$(FINAL_TARGETS):
	$(eval SEED=$(word 3,$(subst -, ,$@)))
	$(eval ALGO=$(word 4,$(subst -, ,$@)))
	$(eval INDEX=$(shell python3 -c "print('$(FINAL_TARGETS)'.split().index('$@'))"))
	@echo "Scheduling $(ALGO) (seed=$(SEED)) with $(INDEX)s delay..."
	@sleep $(INDEX) && $(GLFW_ENV) $(PPY) \
		$(if $(filter ppo,$(ALGO)),$(SCRIPT),$(SCRIPT2)) \
		--env_id="$(ENV)" \
		$(if $(filter ppo,$(ALGO)),--clip_coef=$(CLIP_COEF) --gae_lambda=$(GAE_LAMBDA),--gamma=$(GAMMA) --tau=$(TAU)) \
		--seed=$(SEED) --final-run
