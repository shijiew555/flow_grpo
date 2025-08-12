import torch
from diffusers import FluxKontextPipeline
from flow_grpo.diffusers_patch.flux_kontext_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
from .flux_trainer import FluxTrainer


class FluxKontextTrainer(FluxTrainer):
    """FLUX Kontext model trainer implementation."""
    
    def load_pipeline(self):
        """Load FLUX Kontext pipeline."""
        return FluxKontextPipeline.from_pretrained(
            self.config.pretrained.model, torch_dtype=torch.bfloat16
        )
        
    def setup_data_loader(self):
        """Setup data loader with image dataset for Kontext model."""
        from .datasets import GenevalPromptImageDataset, DistributedKRepeatSampler
        from torch.utils.data import DataLoader
        
        # Use GenevalPromptImageDataset for Kontext model
        dataset = GenevalPromptImageDataset(self.config.dataset, 'train')
        
        sampler = DistributedKRepeatSampler(
            dataset=dataset,
            batch_size=self.config.sample.train_batch_size,
            k=getattr(self.config.sample, 'num_image_per_prompt', self.config.sample.num_images_per_prompt),
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            seed=getattr(self.config.sample, 'seed', 42),
        )
        
        self.dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=1,
            collate_fn=dataset.collate_fn,
        )
        
    def compute_text_embeddings(self, prompts, images=None):
        """Compute FLUX Kontext text embeddings with image support."""
        if images is not None:
            # Handle image inputs for Kontext model
            prompt_embeds, pooled_prompt_embeds, text_ids, image_ids = encode_prompt(
                self.pipeline,
                prompts,
                self.accelerator.device,
                do_classifier_free_guidance=False,
                images=images
            )
            return prompt_embeds, pooled_prompt_embeds, text_ids, image_ids
        else:
            # Fall back to standard text-only embeddings
            return super().compute_text_embeddings(prompts)
            
    def compute_log_prob(self, latents, noise, prompt_embeds, pooled_prompt_embeds, 
                        text_ids, image_ids, timesteps, noise_scheduler, with_grad=False):
        """Compute log probabilities for FLUX Kontext."""
        return pipeline_with_logprob(
            self.pipeline,
            latents=latents,
            noise=noise,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            image_ids=image_ids,
            timesteps=timesteps, 
            noise_scheduler=noise_scheduler,
            with_grad=with_grad
        )