# define variables
IMAGE_NAME  = ft_linear_regression
CONTAINER_NAME = ft_lr_container

# ─────────────────────────────────────
# main commands
# ─────────────────────────────────────

# target by default: build then run
all: build run

# build Docker image
build:
	@echo "🔨 Building Docker image..."
	docker build -t $(IMAGE_NAME) .
	@echo "✅ Image built: $(IMAGE_NAME)"

# run container: train + predict (interactive mode, since predict requires user input)
run:
	@echo "🚀 Running container..."
	docker run -it --rm \
		-v $(PWD):/app \
		--name $(CONTAINER_NAME) $(IMAGE_NAME)

# only run training
train:
	docker run --rm \
		-v $(PWD):/app \
		--name $(CONTAINER_NAME) $(IMAGE_NAME) \
			python3 train.py

# only run prediction (requires interaction)
predict:
	docker run -it --rm \
		-v $(PWD):/app \
		--name $(CONTAINER_NAME) $(IMAGE_NAME) \
			python3 predict.py

# bonus
bonus:
	docker run -it --rm \
		-v $(PWD):/app \
		--name $(CONTAINER_NAME) $(IMAGE_NAME) \
			python3 bonus.py


# ─────────────────────────────────────
# cleanup commands
# ─────────────────────────────────────

# stop and remove container
stop:
	@echo "🛑 Stopping container..."
	docker stop $(CONTAINER_NAME) 2>/dev/null || true
	docker rm   $(CONTAINER_NAME) 2>/dev/null || true

# delete image
clean: stop
	@echo "🧹 Removing image..."
	docker rmi $(IMAGE_NAME) 2>/dev/null || true

# thoroughly clean up (containers + image + thetas.json)
fclean: clean
	@echo "🧹 Removing thetas.json..."
	rm -f thetas.json

# rebuild
re: fclean all

# declare phony targets (not corresponding to real files)
.PHONY: all build run train predict bonus stop clean fclean re
