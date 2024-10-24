#!/bin/bash
echo "file name:    $0"
echo "CUDA:         $1"

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Joker in DC, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt="a DSLR portrait of Kratos in God of War, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True 

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt="a head of I am Groot, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True 

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Vincent van Gogh, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.0008 system.area_relax=True

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Batman, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True   system.loss.lambda_scaling=100.0

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Two-face in DC in Marvel, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True  

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt="a DSLR portrait of Obama, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
system.guidance.guidance_scale=15 system.max_grad=0.001 system.area_relax=True  

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt="a DSLR portrait of Elon Musk, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
system.guidance.guidance_scale=25 trainer.max_steps=10000 system.max_grad=0.001 system.area_relax=True  

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a head of a man with a large afro, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True  

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a head of an alien, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True  

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Gandalf, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True  

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt="a DSLR portrait of Geralt in The Witcher, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True   system.loss.lambda_scaling=100.0

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt="a DSLR portrait of Doctor Strange, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True   system.loss.lambda_scaling=100.0

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt="a head of Hulk, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt="a DSLR portrait of Lionel Messi, masterpiece, Studio Quality, 8k, ultra-HD, next generation" \
system.guidance.guidance_scale=50 system.max_grad=0.001 system.area_relax=True

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a head of Caesar in Rise of the Planet of the Apes, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True  

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Vincent van Gogh, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_dsd=True system.max_grad=0.001 system.area_relax=True  \
system.prompt_processor.negative_prompt="sculpture, statue, shadow, dark face, eyeglass, glasses, noise,pattern, strange color, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions,long neck"

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Salvador Dalí, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True  \
system.prompt_processor.negative_prompt="sculpture, statue, shadow, dark face, eyeglass, glasses, noise,pattern, strange color, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions,long neck"

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Captain America, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a head of Spider Man, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True  

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Dwayne Johnson, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Terracotta Army, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True 

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a head of Thanos in Marvel, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 trainer.max_steps=10000 system.area_relax=True  \
system.prompt_processor.negative_prompt="sculpture, statue, shadow, dark face, eyeglass, glasses, noise,pattern, strange color, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions,long neck"

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Thor in Marvel, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 trainer.max_steps=10000 system.area_relax=True  \
system.prompt_processor.negative_prompt="sculpture, statue, shadow, dark face, eyeglass, glasses, noise,pattern, strange color, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions,long neck"

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Leo Tolstoy, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True  \
system.prompt_processor.negative_prompt="sculpture, statue, shadow, dark face, eyeglass, glasses, noise,pattern, strange color, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions,long neck"

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Saul Goodman, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 trainer.max_steps=10000 system.area_relax=True 

CUDA_VISIBLE_DEVICES=$1 python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a head of Iron Man, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True