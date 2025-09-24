# Exploring Model Editing for LLM-based Aspect-Based Sentiment Classification

[![Paper](https://img.shields.io/badge/Paper-AAAI2025-blue.svg)](https://github.com/your-username/your-repo)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the official implementation for the AAAI 2025 paper "Exploring Model Editing for LLM-based Aspect-Based Sentiment Classification".

## 📋 Abstract

This work investigates model editing to serve as an efficient method for adapting LLMs to solve aspect-based sentiment classification and develops a model editing approach that focuses exclusively on these critical parts of the LLM, leading to a more efficient method for adapting LLMs.

## 🔧 Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU 
- 16GB+ GPU memory for Llama-2-7B model
- Tested on PyReft (v0.0.5) + Pyvene (v0.1.2)


### Model Setup

Download the Llama-2-7B model and update the model path in `model_editing.py`:

```python
model_name_or_path = "/path/to/your/Llama-2-7b-hf/"
```

## 📊 Dataset

The code expects data in the following format:
```json
{
    "sentence": "The food here is amazing but the service is terrible.",
    "aspect": "food",
    "label": "Pos"
}
```

### Data Domains
- **device**: Electronics and gadgets
- **laptop**: Laptop computers  
- **rest**: Restaurants and food
- **service**: Customer service

Place your data files in the `data/` directory with the naming convention:
- `{domain}_train.json`
- `{domain}_test.json`

## 🏃‍♂️ Usage

### Quick Start

To simplify the code, we refactored the original implementation and combined both training and inference into a single file. The code is very straightforward. Run the automated training pipeline:

```bash
bash reft.sh
```

### Manual Training

For custom configurations, run the training script directly:

```bash
python model_editing.py <train_domain> <fs_layer> <fn_layer> <use_lora>
```

**Parameters:**
- `train_domain`: Training domain (`device`, `laptop`, `rest`, `service`)
- `fs_layer`: Starting layer for intervention (e.g., 10)
- `fn_layer`: Ending layer for intervention (e.g., 15)
- `use_lora`(Weight-based Editing): Whether to use Weight-based Editing on o_proj (`true` or `false`)

**Example:**
```bash
python model_editing.py laptop 10 15 true
```

### Configuration Options

#### Layer Configuration
Modify the layer ranges in `reft.sh`:
```bash
readonly FS_LAYER_VALUES=(0 5 10 15 20)
readonly FN_LAYER_VALUES=(31 10 15 20 25)
```

#### LoRA Configuration
Adjust LoRA parameters in `model_editing.py`:
```python
peft_config = LoraConfig(
    r=4,                    # Rank
    lora_alpha=8,          # Scaling factor
    target_modules=["o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

#### REFT Configuration
Modify intervention parameters:
```python
reft_config = pyreft.ReftConfig(representations=[{
    "layer": l,
    "component": "block_output",
    "low_rank_dimension": 4,
    "intervention": pyreft.LoreftIntervention(...)
}])
```

## 📈 Results

Results are saved in JSON format in the `results/` directory with the naming pattern:
`{train_domain}_{test_domain}_{fs_layer}_{fn_layer}_{tag}.json`

### Output Metrics
- **Test Accuracy**: Classification accuracy on test set
- **Training Time**: Time taken for training
- **Inference Time**: Time taken for evaluation  
- **Model Parameters**: Trainable parameter counts and percentages
- **Layer Configuration**: Intervention layer ranges

### Example Output
```json
{
    "trainable_intervention_parameters": 196632,
    "trainable_model_parameters": 196608,
    "all_model_parameters": 6738612224,
    "trainable_percentage": 0.006,
    "train_time": 98.029,
    "inference_time": 67.069,
    "test_acc": 0.962,
    "trasformer_layers": [
        10,
        11,
        12,
        13,
        14,
        15
    ],
    "train_data_path": "-",
    "test_data_path": "-",
    "lr": "2e-05"
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `training_args`
   - Use gradient checkpointing
   - Try smaller layer ranges

2. **Model Path Error**
   - Verify the Llama-2-7B model path in `model_editing.py`
   - Ensure model files are properly downloaded

3. **Data Format Error**
   - Check JSON format matches expected schema
   - Verify file paths in data directory

4. **Currently, this project only supports single-GPU execution.**

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{li2025ME-ABSC,
  title={Exploring Model Editing for LLM-based Aspect-Based Sentiment Classification},
  author={Li, Shichen and Wang, Zhongqing and Zhao, Zheyu and Zhang, Yue and Li, Peifeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}，
  url={https://ojs.aaai.org/index.php/AAAI/article/view/34625}
}

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyReft](https://github.com/stanfordnlp/pyreft) for the REFT implementation
- [PEFT](https://github.com/huggingface/peft) for LoRA support
- [Transformers](https://github.com/huggingface/transformers) for model implementations
- [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) base model

---

**Note**: This repository contains the code for research purposes. Please ensure you comply with the respective model licenses when using pre-trained models.

