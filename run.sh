
set -e

IMAGE_NAME="llm_predictor"
IMAGE_TAG="latest"

echo "#####################################################"
echo "### Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "#####################################################"

docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo ""
echo "#####################################################"
echo "### Running experiment in Docker container...     ###"
echo "#####################################################"

# Create cache folder on host (outside project)
mkdir -p ~/.cache/llm_predictor

docker run \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/Data:/workspace/Data" \
  -v "$(pwd)/results:/workspace/results" \
  -v "$HOME/.cache/llm_predictor:/workspace/.cache" \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  -e TORCH_HOME=/workspace/.cache/torch \
  -e MPLCONFIGDIR=/workspace/.config/matplotlib \
  -e TMPDIR=/tmp \
  -e TEMP=/tmp \
  -e TMP=/tmp \
  --rm \
  "${IMAGE_NAME}:${IMAGE_TAG}" \
  python multi_ticker_experiment.py

echo ""
echo "#####################################################"
echo "### Experiment completed. Results in ./results folder"
echo "#####################################################"

