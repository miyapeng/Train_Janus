# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from transformers.configuration_utils import PretrainedConfig

from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector


class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))

    def forward(self, 
                input_ids, input_text_ids, input_image_ids, output_image_ids, labels=None, task="understanding", return_dict=True, **kwargs):
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"task: {task}")
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
        if task == "understanding":
            return super().forward(input_ids, labels, **kwargs)
        
        elif task == "generation":
            image_token_num_per_image = 576
            cfg_weight = 5
            temperature = 1
            # print("+++++++++++++++++++++++++++++++++++++++++++++++")
            # print(f"input_ids_0: {input_ids.size(0)}")
            # print(f"input_ids_1: {input_ids.size(1)}")
            # print("+++++++++++++++++++++++++++++++++++++++++++++++")
            tokens = torch.zeros((2*input_ids.size(0), input_ids.size(1)), dtype=torch.int).cuda()
            for i in range(2):
                tokens[i*input_ids.size(0):(i+1)*input_ids.size(0), :] = input_ids
                if i % 2 != 0:
                    tokens[i*input_ids.size(0):(i+1)*input_ids.size(0), 1:-1] = 100015 # pad_id

            inputs_embeds = self.language_model.get_input_embeddings()(tokens)
            print("Embedding size:", self.language_model.get_input_embeddings().weight.size(0))
            print("Max token id in input_ids:", input_ids.max())
            
            print("+++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"inputs_embeds_shape: {inputs_embeds.shape}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++")
            outputs = self.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None)

            hidden_states = outputs.last_hidden_state
            logits = self.gen_head(hidden_states)

            logits_cond = logits[0::2, :]
            logits_uncond = logits[1::2, :]

            all_logits = logits_uncond + cfg_weight * (logits_cond - logits_uncond)
            # #Show image to see
            # # 截取后 576 个位置，对应 image token 预测
            # image_logits = all_logits[:, -576:, :]  # [1, 576, 16384]
            # print(f"image_logits_shape: {image_logits.shape}")
            # probs = torch.softmax(image_logits / temperature, dim=-1)
            # print(f"probs_shape:{probs.shape}")
            # #print(probs[0,0:5,0:10])
            # B, N, V = probs.shape
            # image_token_ids = torch.multinomial(probs.view(-1, V), num_samples=1).view(B, N)
            # print(f"image_token_ids_shape:{image_token_ids.shape}")
            # print(image_token_ids)
            # # decode 成图像
            # decoded_images = self.gen_vision_model.decode_code(
            #     image_token_ids.to(dtype=torch.int),
            #     shape=[1, 8, 24, 24],  # 通常 384x384 = 24x24
            # )
            # decoded_images = decoded_images.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            # import numpy as np
            # decoded_images = np.clip((decoded_images + 1) / 2 * 255, 0, 255).astype(np.uint8)
            # print(f"decoded_images_shape:{decoded_images.shape}")
            # visual_img = np.zeros((1,384,384,3), dtype=np.uint8)
            # visual_img[:,:,:] = decoded_images

            # # 保存图片
            # import os, PIL.Image
            # os.makedirs("t2i_generated_images", exist_ok=True)
            # for i in range(1):
            #     PIL.Image.fromarray(visual_img[i]).save(f"t2i_generated_images/img_{i}.jpg")

            loss_fct = CrossEntropyLoss()
            shift_logits = all_logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, self.config.gen_head_config.params.image_token_size)

            if labels is not None:
                shift_labels = labels[..., 1:].contiguous()
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            else:
                loss = None
            if not return_dict:
                output = (logits,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
        elif task == "TI2I_generation":
            image_token_num_per_image = 576
            cfg_weight = 5
            temperature = 1
            
            # train without cfg
            device = input_ids.device
            tokens = input_text_ids.to(dtype=torch.int, device=device)
            text_inputs_embeds = self.language_model.get_input_embeddings()(tokens)
            input_image_embeds = self.prepare_gen_img_embeds(input_image_ids)
            output_image_embeds = self.prepare_gen_img_embeds(output_image_ids)
            inputs_embeds = torch.cat([text_inputs_embeds,input_image_embeds,output_image_embeds],dim=1)
            print("+++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"text_inputs_embeds_shape: {text_inputs_embeds.shape}")
            print(f"input_image_embeds_shape: {input_image_embeds.shape}")
            print(f"output_image_embeds_shape: {output_image_embeds.shape}")
            print(f"inputs_embeds_shape: {inputs_embeds.shape}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++")

            # #train with cfg
            # tokens = torch.zeros((2*input_text_ids.size(0), input_text_ids.size(1)), dtype=torch.int).cuda()
            # for i in range(2):
            #     tokens[i*input_text_ids.size(0):(i+1)*input_text_ids.size(0), :] = input_text_ids
            #     if i % 2 != 0:
            #         tokens[i*input_text_ids.size(0):(i+1)*input_text_ids.size(0), 1:-1] = 100015 # pad_id
            # inputs_text_embeds = self.language_model.get_input_embeddings()(tokens)

            # input_image_embeds = self.prepare_gen_img_embeds(input_image_ids)
            # output_image_embeds = self.prepare_gen_img_embeds(output_image_ids)

            # pad_embed = self.language_model.get_input_embeddings()(torch.tensor([100015]).cuda()).squeeze(0)  # [D]

            # input_image_embeds_uncond = pad_embed.unsqueeze(0).unsqueeze(0).expand(input_image_embeds.size(0), input_image_embeds.size(1), -1)   # [B, T1, D]
            # output_image_embeds_uncond = pad_embed.unsqueeze(0).unsqueeze(0).expand(input_image_embeds.size(0), output_image_embeds.size(1), -1) # [B, T2, D]

            # input_image_embeds = torch.cat([input_image_embeds, input_image_embeds_uncond], dim=0)     # [2B, T1, D]
            # output_image_embeds = torch.cat([output_image_embeds, output_image_embeds_uncond], dim=0)  # [2B, T2, D]

            # inputs_embeds = torch.cat([inputs_text_embeds, input_image_embeds, output_image_embeds], dim=1)  # [2B, L+T1+T2, D]

            # print("+++++++++++++++++++++++++++++++++++++++++++++++")
            # print(f"text_inputs_embeds_shape: {inputs_text_embeds.shape}")
            # print(f"input_image_embeds_shape: {input_image_embeds.shape}")
            # print(f"output_image_embeds_shape: {output_image_embeds.shape}")
            # print(f"inputs_embeds_shape: {inputs_embeds.shape}")
            # print("+++++++++++++++++++++++++++++++++++++++++++++++")

            outputs = self.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None)

            hidden_states = outputs.last_hidden_state
            logits = self.gen_head(hidden_states)
            print(f"logits_shape:{logits.shape}")

            # # cfg compute
            # logits_cond = logits[0::2, :]
            # logits_uncond = logits[1::2, :]
            # logits = logits_uncond + cfg_weight * (logits_cond - logits_uncond)

            logits = logits.float() #turn float16 into float32
            print(f"logits_type: {logits.dtype}")

            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, self.config.gen_head_config.params.image_token_size)

            if labels is not None:
                shift_labels = labels[..., 1:].contiguous()
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

                # print(f"shift_labels_shape: {shift_labels.shape}")
                # print(shift_labels[612:622])
                # print(f"shift_logits_shape: {shift_logits.shape}")
                # print(shift_logits[612,:622])
            else:
                loss = None
            if not return_dict:
                output = (logits,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
        elif task == "generation_direct":
            outputs = self.language_model.model(input_ids=input_ids, **kwargs)
            hidden_states = outputs[0] # possibly outputs[0]
            logits = self.gen_head(hidden_states)

            loss = None
            
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, self.config.gen_head_config.params.image_token_size)
                
            if labels is not None:  
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            else:
                loss = None

            if not return_dict:
                output = (logits,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output
            
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
