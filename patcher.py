class PULIDModelPatcher():
    def patch_model(model, pulid, *args, **kwargs):
        model.pulid = pulid

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
            self.model.diffusion_model.ca_idx = 0
            for i, block in enumerate(self.model.diffusion_model.double_blocks):
                block.pulid_index = i
                block.pulid = self.pulid
                block.model = self.model.diffusion_model
                
            for i, block in enumerate(self.model.diffusion_model.single_blocks):
                block.pulid_index = i
                block.pulid = self.pulid
                block.model = self.model.diffusion_model
                
            return self.model.diffusion_model.original_forward(*args, **kwargs)
        
        for i, block in enumerate(model.model.diffusion_model.double_blocks):
            if not hasattr(block, 'original_forward'):
                block.original_forward = block.forward
            block.forward = new_double_forward.__get__(block)
            
        for i, block in enumerate(model.model.diffusion_model.single_blocks):
            if not hasattr(block, 'original_forward'):
                block.original_forward = block.forward
            block.forward = new_single_forward.__get__(block)
            
        if not hasattr(model.model.diffusion_model, 'original_forward'):
            model.model.diffusion_model.original_forward = model.model.diffusion_model.forward
            
        model.model.diffusion_model.forward = new_model_forward.__get__(model)

    def unpatch_model(model, device_to=None, unpatch_weights=True):
        for block in model.model.diffusion_model.double_blocks:
            if hasattr(block, 'original_forward'):
                block.forward = block.original_forward
                delattr(block, 'original_forward')
                
        for block in model.model.diffusion_model.single_blocks:
            if hasattr(block, 'original_forward'):
                block.forward = block.original_forward
                delattr(block, 'original_forward')
                
        if hasattr(model.model, 'original_forward'):
            model.model.diffusion_model.forward = model.model.diffusion_model.original_forward
            delattr(model.model, 'original_forward')
            
        if hasattr(model, 'pulid'):
            delattr(model, 'pulid')
 
 