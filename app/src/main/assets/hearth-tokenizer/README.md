# Bundled CLIP tokenizer assets

These files are the standard OpenAI CLIP ViT-L/14 BPE vocabulary and merge rules used by Hearth to turn prompts into token IDs. They are packaged with Ponko so normal SD 1.5 and SDXL checkpoint imports can load offline without asking users to choose tokenizer files.

Original model repository: https://huggingface.co/openai/clip-vit-large-patch14
- vocab.json: https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/vocab.json
- merges.txt: https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/merges.txt

A GitHub mirror was used to vendor the text files into this repository. The CLIP project source is available under the MIT license: https://github.com/openai/CLIP/blob/main/LICENSE

These assets are tokenizer metadata, not model weights. Custom model-set tokenizer slots may override the bundled files.
