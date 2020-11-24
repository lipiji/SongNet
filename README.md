# SongNet

```
@inproceedings{li-etal-2020-rigid,
    title = "Rigid Formats Controlled Text Generation",
    author = "Li, Piji and Zhang, Haisong and Liu, Xiaojiang and Shi, Shuming",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.68",
    doi = "10.18653/v1/2020.acl-main.68",
    pages = "742--751"
}
```

### Run
- python prepare_data.py
- ./train.sh 

### Evaluation
- Modify test.py: m_path = the best dev model
- ./test.sh
- python metrics.py

### Polish
- ./polish.sh

### Download
- The pretrained Chinese Language Model: https://drive.google.com/file/d/1g2tGyUwPe86vPn2nub1vkQva5lwtZ6Rd/view 
- The finetuned SongCi model: https://drive.google.com/file/d/16A2AzuU7slf7xj2QdLcBAorUCCaCk650/view

#### Reference

- Guyu: https://github.com/lipiji/Guyu
