export MODEL_ID="SG161222/Realistic_Vision_V2.0" # change this
export SAFETY_MODEL_ID="CompVis/stable-diffusion-safety-checker"
export IS_FP16=1
export USERNAME="purrfectai" # change this
export REPLICATE_MODEL_ID="rv_2.0_upscale" # change this

echo "MODEL_ID=$MODEL_ID" > .env
echo "SAFETY_MODEL_ID=$SAFETY_MODEL_ID" >> .env
echo "IS_FP16=$IS_FP16" >> .env

python script/download-weights.py
cog run python test.py --test_img2img --test_text2img --test_adapter
cog push r8.im/$USERNAME/$REPLICATE_MODEL_ID
