<h1 align="center"> <p>PEFT with LoRAN 📻</p></h1>
<h3 align="center">
    <p>Low-Rank Adaptation with Non-Linear Transformation (LoRAN)</p>
</h3>

Parameter-efficient fine-tuning (PEFT) has been widely used in adapting LLMs to downstream tasks, and low-rank adaptation (LoRA) is one of the popular methods in it. In this study, we improve LoRA approaches to alleviate the performance loss from the gap between constrained adaptations and complex downstream tasks by introducing a non-linear transformation (call it LoRAN). For a better adaptation, we also design a brand-new activation function to appropriately fit the accumulated gradient updates during fine-tuning. We implement our method with multiple advanced large language models. Experimental results show that it significantly outperforms a strong baseline on SAMSum summarization and 20 Newsgroups classification tasks. Moreover, when a lower rank is applied, our approach even yields a 1.95-point improvement in the text classification task. 

In this project, we develop our LoRAN based on the open-sourced PEFT developed by Huggingface. Therefore, implementations with the original PEFT can easily upgrade their codes to apply our LoRAN in their project. 

## Getting started

This project is developed based on PEFT version 0.4.0. Please install it and replace the `src` with ours. 

### How to use LoRAN

The use of LoRAN is very similar to LoRA in PEFT. What you need to do is replace introducing `LoraConfig` with `MLoraConfig` . The setting of `MLoraConfig` is simple. Here are the arguments:

| Arguments                                 | Type                 | Description                                                  |
| ----------------------------------------- | -------------------- | ------------------------------------------------------------ |
| <font color=red>r</font>                  | List[int]            | LoRAN attention dimension.                                   |
| <font color=red>target_modules</font>     | Union[List[str],str] | The names of the modules to apply LoRAN to.                  |
| <font color=red>mlora_alpha</font>        | int                  | The alpha parameter for LoRAN scaling.                       |
| <font color=red>mlora_dropout</font>      | float                | The dropout probability for LoRAN layers.                    |
| <font color=red>mlora_af</font>           | str                  | The activation function for LoRAN layers.                    |
| <font color=red>mlora_af_sin_A</font>     | float                | The A parameter for the Sin function used in LoRAN layers (Only works when mlora_af is "Sin"). |
| <font color=red>mlora_af_sin_omega</font> | float                | The omega parameter for the Sin function used in LoRAN layers (Only works when mlora_af is "Sin"). |
| mlora_use_P                               | bool                 | Set this to True if Ps are used in LoRAN.                    |
| mlora_use_b                               | bool                 | Set this to True if bs are used in LoRAN.                    |
| fan_in_fan_out                            | bool                 | Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`. |
| bias                                      | str                  | Bias type for LoRAN. Can be 'none', 'all' or 'mlora_only'    |
| modules_to_save                           | List[str]            | List of modules apart from LoRAN layers to be set as trainable and saved in the final checkpoint. |
| layers_to_transform                       | Union[List[int],int] | The layer indexes to transform, if this argument is specified, it will apply the LoRAN transformations on the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRAN transformations on the layer at this index. |
| layers_pattern                            | str                  | The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer pattern is not in the common layers pattern. |

The red arguments are directly related to LoRAN, and here is an example of  `MLoraConfig` in LLMs:

```python
peft_config = MLoraConfig(
            r=[64,4096],
            mlora_alpha=16,
            mlora_af="Sin",
            mlora_af_sin_A=5e-5,
            mlora_af_sin_omega=1e4,
            mlora_use_P=False,
            mlora_use_b=False,
            mlora_dropout=0.2,
            bias="none",
            task_type="CAUSAL_LM",
        )
```

### Experiments

We test our approach on SAMSum[1] and 20 Newsgroups[2] tasks with standard pre-processing. Llama-2-Large[3] is used as the foundation model. All these fine-tuned adapters are stored in `experiment`. Feel free to try them 🎉.** 

Due to the limitation of Git platform, we have to split the big file into several pieces. Please merge them back when you use, like this:

```shell
cd experiment/SAMSum/Llama-2-7b/LoRAN
cat adapter_model.bin.* > adapter_model.bin
```

### Reference

*[1] Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. 2019. SAMSum corpus: A human-annotated dialogue dataset for abstractive summarization. In Proceedings of the 2nd Workshop on New Frontiers in Summarization, pages 70–79, Hong Kong, China. Association for Computational Linguistics.* 

*[2] Ken Lang. 1995. Newsweeder: Learning to filter net-news. In Machine Learning, Proceedings of the Twelfth International Conference on Machine Learning, Tahoe City, California, USA, July 9-12, 1995,417 pages 331–339. Morgan Kaufmann.*

*[3] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288.*