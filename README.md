# AIRI_2025_Chem_AI

# CoPRA: Улучшенный метод предсказания сродства белок-РНК

## Обзор
Репозиторий содержит модифицированную версию метода CoPRA, предназначенного для предсказания сродства в комплексах белок-РНК. Основные изменения сосредоточены в файле `CoPRA_pair_encoder.py`.
Было добавлено определение вторичной структуры РНК, а также извлечение координат Cβ и Cγ атомов аминокислотных остатков.

**Оригинальный CoPRA**:  
[https://github.com/hanrthu/CoPRA/tree/main](https://github.com/hanrthu/CoPRA/tree/main) В частности, был модифицирован скрипт `models\components\coformer.py`.

**Оригинальная статья**:  
@article{han2024copra,
  title={CoPRA: Bridging Cross-domain Pretrained Sequence Models with Complex Structures for Protein-RNA Binding Affinity Prediction},
  author={Han, Rong and Liu, Xiaohong and Pan, Tong and Xu, Jing and Wang, Xiaoyu and Lan, Wuyang and Li, Zhenyu and Wang, Zixuan and Song, Jiangning and Wang, Guangyu and others},
  journal={arXiv preprint arXiv:2409.03773},
  year={2024}
}
