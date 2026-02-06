# .bazelrc.ml – ML build optimizations with CUDA support

build:ml --config=ml
build:ml --copt=-march=native
build:ml --linkopt=-march=native
build:ml --define=USE_CUDA=1
build:ml --action_env=CUDA_VISIBLE_DEVICES=0,1
build:ml --jobs=auto
build:ml --define=tensorflow_enable_cuda=1
build:ml --define=grpc_no_ares=true  # fix gRPC DNS issues on some systems

test:ml --test_output=errors
run:ml --test_output=errors

# Remote execution (RBE) for GPU workers
build:rbe --remote_executor=grpcs://remotebuild.googleapis.com
build:rbe --remote_instance_name=projects/your-project/locations/us-west1/instances/default_instance
build:rbe --google_default_credentials
