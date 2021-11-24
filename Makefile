.PHONY: help docker-build
.DEFAULT: help

tag=latest
version=`python setup.py --version`

help:
	@echo "Premade recipes"
	@echo
	@echo "make docker-build [tag=TAG]"
	@echo "\tBuilds a docker image from source. Defaults to 'fmriprep' tag."


docker-build:
	@echo "Building nipreps/fmriprep-rodents:$(tag) with version=$(version)"
	docker build --rm -t nipreps/fmriprep-rodents:$(tag) \
	--build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
	--build-arg VCS_REF=`git rev-parse --short HEAD` \
	--build-arg VERSION=$(version) .
