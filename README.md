# Alternatives 
[ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux) Another dev's implementation seems to further ahead with the implementation, still a direct model patch, but integrates better with comfy backend, recommend trying it out instead.

# Comfy PuLID FLUX Experiments

THIS REPO IS JUST AN EXPERIMENT TO TRY TO GET IT WORKING.

It's only tested with GGUF on my Macbook M3, if it works/doesn't work on other configs/setups we will find out i guess.

This is not the way to do this properly, Cubiq is working on a real verson, and I believe 
the comfy team are working on hooks to improve the way we can integrate into the pipeline for new forward methods hopefully. 

![Example Workflow with GGUF and PuLID](example.png)

### Information

This is a mess, please don't use this code it's not really ready to be used publicly but people asked me for a look at it so here it is.
Lots of things are hardcoded, (dtype, device, etc), so if your going to use it to play around please take a moment and look through the code. 

I did this half asleep trying to figure out whats what in the comfy codebase and I don't primarily code in python so I likely didn't do things memory efficiently etc. 

I still don't grasp the modelpatcher from comfy, and don't know the hooks that are available anyway, so right now it a very dirty monkey patch of the diffusion model 
to brute force the pipeline to process pulid.

### Issues

1. This is really a brute force of the model, so if your loading the model and it goes into the patcher, that's now a pulid model, 
it breaks the original model node (maybe someone can fix it maybe i'm cloning wrong) so you can't use that same model loader in a different ksampler 
currently you'd need to load the model in a new loader.

2. I haven't tested this on standard checkpoint loader, it should work i think, but i've only tested with GGUF so far.

3. Not using the helper functions as should be done (comfy.model_management, etc), so dtypes devices etc are probably not right.

4. I haven't implemented the uncond_id embedding yet so ... ya

5. Likely many more issues i haven't found yet, but i'm still learning comfy's backend and the models in general.

### Performance & Requirements

1.  VRAM: ... A Lot? Didn't test it runs on a macbook with 32gb so... with Q8 and Pulid i was sitting at ~30gb of ram usage, didn't benchmark.
2.  Speed: ... It runs... 30s/it at 1024x1024 on a Macbook M3 32gb, using Q8 GGUF with Hyper 8 step embedded.
   
### Contributing

Like I said this is a for-fun repo that i'm just playing with while the major devs at comfy (and cubiq) work on proper integrations, but by all means if you see a way to 
improve this shoot a PR and i'll try to review/approve.

### Responsible Usage

I don't take any responsibility for your usage of this, use it responsibly and at your own risk, if it doesn't work or blows up the world somehow it's not my fault :P 

Parts of this code were used from Cubiq and from the original project so by all means, all rights to them, the flux team and everyone else that worked on the various projects.


## Resources
[Original PuLID Project](https://github.com/ToTheBeginning/PuLID) from ToTheBeginnning
[The Real Pulid Comfy Extension](https://github.com/cubiq/PuLID_ComfyUI) from Cubiq