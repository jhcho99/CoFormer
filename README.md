# [CVPR'22] Collaborative Transformers for Grounded Situation Recognition
[Paper](https://arxiv.org/abs/2203.16518) | [Model Checkpoint](https://drive.google.com/u/1/uc?id=1EoJ081eEcfp0u-B1qkkA_EX8U3Ko8P3u&export=download&confirm=t) 

- This is the official PyTorch implementation of [Collaborative Transformers for Grounded Situation Recognition](https://arxiv.org/abs/2203.16518). 
- **CoFormer** (**Co**llaborative Glance-Gaze Trans**Former**) achieves state-of-the-art accuracy in every evaluation metric on the SWiG dataset.
- This repository contains instructions, code and model checkpoint.

![prediction_results](https://user-images.githubusercontent.com/55849968/160762073-9a458795-03c1-4b2a-8945-2187b5a48aca.png)

---

## Overview
Grounded situation recognition is the task of predicting the main activity, entities playing certain roles within the activity, and bounding-box groundings of the entities in the given image. To effectively deal with this challenging task, we introduce a novel approach where the two processes for activity classification and entity estimation are interactive and complementary. To implement this idea, we propose **Co**llaborative Glance-Gaze Trans**Former** (CoFormer) that consists of two modules: Glance transformer for activity classification and Gaze transformer for entity estimation. Glance transformer predicts the main activity with the help of Gaze transformer that analyzes entities and their relations, while Gaze transformer estimates the grounded entities by focusing only on the entities relevant to the activity predicted by Glance transformer. Our CoFormer achieves the state of the art in all evaluation metrics on the SWiG dataset.

![overall_architecture](https://user-images.githubusercontent.com/55849968/160762199-def33a41-b333-41c8-b367-7b6c814b987c.png)
Following conventions in the literature, we call an activity ***verb*** and an entity ***noun***. Glance transformer predicts a verb with the help of Gaze-Step1 transformer that analyzes nouns and their relations by leveraging role features, while Gaze-Step2 transformer estimates the grounded nouns for the roles associated with the predicted verb. Prediction results are obtained by feed forward networks (FFNs). 

## Environment Setup
We provide instructions for environment setup.
```bash
# Clone this repository and navigate into the repository
git clone https://github.com/jhcho99/CoFormer.git    
cd CoFormer                                          

# Create a conda environment, activate the environment and install PyTorch via conda
conda create --name CoFormer python=3.9              
conda activate CoFormer                             
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge 

# Install requirements via pip
pip install -r requirements.txt                   
```

## SWiG Dataset
Annotations are given in JSON format, and annotation files are under "SWiG/SWiG_jsons/" directory. Images can be downloaded [here](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip). Please download the images and store them in "SWiG/images_512/" directory.

<p align="left">  
<img src="https://user-images.githubusercontent.com/55849968/160765697-fbd65b24-6cdd-4b57-8a68-0746cfb7c44e.png" height="200">  
</p> 

In the SWiG dataset, each image is associated with Verb, Frame and Groundings.       

A) Verb: each image is paired with a verb. In the annotation file, "verb" denotes the salient action for an image.            

B) Frame: a frame denotes the set of semantic roles for a verb. For example, the frame for verb "Drinking" denotes the set of semantic roles "Agent", "Liquid", "Container" and "Place". In the annotation file, "frames" show the set of semantic roles for a verb, and noun annotations for each role. There are three noun annotations for each role, which are given by three different annotators.        

C) Groundings: each grounding is described in [x1, y1, x2, y2] format. In the annotation file, "bb" denotes bounding-box groundings for roles. Note that nouns can be labeled without groundings, e.g., in the case of occluded objects. When there is no grounding for a role, [-1, -1, -1, -1] is given.       

```
# an example of annotation for an image

"drinking_235.jpg": {
    "verb": "drinking",
    "height": 512, 
    "width": 657, 
    "bb": {"agent": [0, 1, 654, 512], 
           "liquid": [128, 273, 293, 382], 
           "container": [111, 189, 324, 408],
           "place": [-1, -1, -1, -1]},
    "frames": [{"agent": "n10787470", "liquid": "n14845743", "container": "n03438257", "place": ""}, 
               {"agent": "n10129825", "liquid": "n14845743", "container": "n03438257", "place": ""}, 
               {"agent": "n10787470", "liquid": "n14845743", "container": "n03438257", "place": ""}]
    }
```

In imsitu_space.json file, there is additional information for verb and noun.

```
# an example of additional verb information

"drinking": {
    "framenet": "Ingestion", 
    "abstract": "the AGENT drinks a LIQUID from a CONTAINER at a PLACE", 
    "def": "take (a liquid) into the mouth and swallow", 
    "order": ["agent", "liquid", "container", "place"], 
    "roles": {"agent": {"framenet": "ingestor", "def": "The entity doing the drink action"},
              "liquid": {"framenet": "ingestibles", "def": "The entity that the agent is drinking"}
              "container": {"framenet": "source", "def": "The container in which the liquid is in"}        
              "place": {"framenet": "place", "def": "The location where the drink event is happening"}}
    }
```

```
# an example of additional noun information

"n14845743": {
    "gloss": ["water", "H2O"], 
    "def": "binary compound that occurs at room temperature as a clear colorless odorless tasteless liquid; freezes into ice below 0 degrees centigrade and boils above 100 degrees centigrade; widely used as a solvent"
    }
```

#### Additional Details
- All images should be under "SWiG/images_512/" directory.
- train.json file is for train set.
- dev.json file is for development set.
- test.json file is for test set.

## Training
To train CoFormer on a single node with 4 GPUs for 40 epochs, run:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
           --backbone resnet50 --batch_size 16 --dataset_file swig --epochs 40 \
           --num_workers 4 --num_glance_enc_layers 3 --num_gaze_s1_dec_layers 3 \
           --num_gaze_s1_enc_layers 3 --num_gaze_s2_dec_layers 3 --dropout 0.15 --hidden_dim 512 \
           --output_dir CoFormer
```
To train CoFormer on a Slurm cluster with submitit using 4 RTX 3090 GPUs for 40 epochs, run:
```bash
python run_with_submitit.py --ngpus 4 --nodes 1 --job_dir CoFormer \
        --backbone resnet50 --batch_size 16 --dataset_file swig --epochs 40 \
        --num_workers 4 --num_glance_enc_layers 3 --num_gaze_s1_dec_layers 3 \
        --num_gaze_s1_enc_layers 3 --num_gaze_s2_dec_layers 3 --dropout 0.15 --hidden_dim 512 \
        --partition rtx3090
```

- A single epoch takes about 45 minutes. Training CoFormer for 40 epochs takes around 30 hours on a single machine with 4 RTX 3090 GPUs.          
- We use AdamW optimizer with learning rate 10<sup>-4</sup> (10<sup>-5</sup> for backbone), weight decay 10<sup>-4</sup> and β = (0.9, 0.999).    
    - Those learning rates are divided by 10 at epoch 30.
- Random Color Jittering, Random Gray Scaling, Random Scaling and Random Horizontal Flipping are used for augmentation.

## Evaluation
To evaluate CoFormer on the dev set with the saved model, run:
```bash
python main.py --saved_model CoFormer_checkpoint.pth --output_dir CoFormer --dev
```
To evaluate CoFormer on the test set with the saved model, run:
```bash
python main.py --saved_model CoFormer_checkpoint.pth --output_dir CoFormer --test
```
- Model checkpoint can be downloaded [here](https://drive.google.com/u/1/uc?id=1EoJ081eEcfp0u-B1qkkA_EX8U3Ko8P3u&export=download&confirm=t).

## Inference
To run an inference on a custom image, run:
```bash
python inference.py --image_path inference/filename.jpg \
                    --saved_model CoFormer_checkpoint.pth \
                    --output_dir inference
```

## Results
We provide several experimental results.

![quantitative](https://user-images.githubusercontent.com/55849968/160766858-c769d713-9114-45ea-9c09-efbbbef2b8d3.png)
![qualitative_1](https://user-images.githubusercontent.com/55849968/160766961-e0cf0f2c-296c-4694-b80a-5d55577dbc2f.png)
![qualitative_2](https://user-images.githubusercontent.com/55849968/160767034-a4342caf-aed1-4850-8597-7d55b89adea0.png)

## Our Previous Work
We proposed **GSRTR** for this task using a simple transformer encoder-decoder architecture:
- [Grounded Situation Recognition with Transformers](https://github.com/jhcho99/gsrtr)

## Acknowledgements
Our code is modified and adapted from these amazing repositories:
- [End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr)          
- [Grounded Situation Recognition](https://github.com/allenai/swig)

## Contact
Junhyeong Cho (jhcho99.cs@gmail.com)

## Citation
If you find our work useful for your research, please cite our paper:

````BibTeX
@InProceedings{cho2022CoFormer,
    title={Collaborative Transformers for Grounded Situation Recognition},
    author={Junhyeong Cho and Youngseok Yoon and Suha Kwak},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
````

## License
CoFormer is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.
