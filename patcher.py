from comfy.model_patcher import ModelPatcher, copy
 
class PULIDModelPatcher(ModelPatcher):
    def __init__(self, model, load_device, offload_device, size, weight_inplace_update: bool=False, pulid=None):
        super().__init__(model, load_device, offload_device, size, weight_inplace_update)
        self.pulid = pulid

    def clone(self, pulid=None):
        n = PULIDModelPatcher(self.model, self.load_device, self.offload_device, self.size, weight_inplace_update=self.weight_inplace_update, pulid=pulid)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid
        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        return n

    def patch_model(self, *args, **kwargs):
        self.model.pulid = self.pulid

        def new_double_forward(self, img, txt, vec, pe):
            if not hasattr(self, 'original_forward'):
                raise AttributeError("original_forward method not found. Make sure it's properly set.")
            
            original_img, original_txt = self.original_forward(img, txt, vec, pe)
            if hasattr(self.pulid, 'id_embedding') and self.pulid_index % self.pulid.double_interval == 0:
                id_embed = self.pulid.id_embedding
                id_weight = self.pulid.id_weight
                
                img_cond = original_img + id_weight * self.pulid.pulid_ca[self.model.ca_idx](id_embed, original_img)
                
                self.model.ca_idx += 1
                return (img_cond, original_txt)
            return (original_img, original_txt)

        def new_single_forward(self, x, vec, pe):
            if not hasattr(self, 'original_forward'):
                raise AttributeError("original_forward method not found. Make sure it's properly set.")
            
            original_img = self.original_forward(x, vec, pe)
            if hasattr(self.pulid, 'id_embedding') and self.pulid_index % self.pulid.single_interval == 0:
                id_embed = self.pulid.id_embedding
                id_weight = self.pulid.id_weight
                
                img_cond = original_img + id_weight * self.pulid.pulid_ca[self.model.ca_idx](id_embed, original_img)
                
                self.model.ca_idx += 1
                return img_cond
            return original_img

        def new_model_forward(self, *args, **kwargs):
            self.diffusion_model.ca_idx = 0
            for i, block in enumerate(self.diffusion_model.double_blocks):
                block.pulid_index = i
                block.pulid = self.pulid
                block.model = self.diffusion_model
                
            for i, block in enumerate(self.diffusion_model.single_blocks):
                block.pulid_index = i
                block.pulid = self.pulid
                block.model = self.diffusion_model
                
            return self.diffusion_model.original_forward(*args, **kwargs)
        
        for i, block in enumerate(self.model.diffusion_model.double_blocks):
            if not hasattr(block, 'original_forward'):
                block.original_forward = block.forward
            block.forward = new_double_forward.__get__(block)
            
        for i, block in enumerate(self.model.diffusion_model.single_blocks):
            if not hasattr(block, 'original_forward'):
                block.original_forward = block.forward
            block.forward = new_single_forward.__get__(block)
            
        if not hasattr(self.model.diffusion_model, 'original_forward'):
            self.model.diffusion_model.original_forward = self.model.diffusion_model.forward
            
        self.model.diffusion_model.forward = new_model_forward.__get__(self.model)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        for block in self.model.diffusion_model.double_blocks:
            if hasattr(block, 'original_forward'):
                block.forward = block.original_forward
                delattr(block, 'original_forward')
                
        for block in self.model.diffusion_model.single_blocks:
            if hasattr(block, 'original_forward'):
                block.forward = block.original_forward
                delattr(block, 'original_forward')
                
        if hasattr(self.model, 'original_forward'):
            self.model.diffusion_model.forward = self.model.diffusion_model.original_forward
            delattr(self.model, 'original_forward')
            
        if hasattr(self.model, 'pulid'):
            delattr(self.model, 'pulid')
 