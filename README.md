# GlobalMind: Global multi-head interactive self-attention network for hyperspectral change detection
Pytorch implementation of ISPRS paper "GlobalMind: Global multi-head interactive self-attention network for hyperspectral change detection".

Abstract: High spectral resolution imagery of the Earth’s surface enables users to monitor changes over time in fine- grained scale, playing an increasingly important role in agriculture, defense, and emergency response. However, most current algorithms are still confined to describing local features and fail to incorporate a global perspective, which limits their ability to capture interactions between global features, thus usually resulting in incomplete change regions. In this paper, we proposed a **Global Multi-head INteractive selfattention change Detection network (GlobalMind)** to explore the implicit correlation between different surface objects and variant land cover transformations, acquiring a comprehensive understanding of the data and accurate change detection result. Firstly, a simple but effective **Global Axial Segmentation (GAS)** is designed to expand the self-attention computation along the row space or column space of hyperspectral images, allowing the global connection with high efficiency. Secondly, with GAS, the global spatial multi-head interactive self-attention (**GlobalM**) module is crafted to mine the abundant spatial-spectral feature involving potential correlations between the ground objects from the entire rich and complex hyperspectral space. Moreover, to acquire the accurate and complete crosstemporal changes, we devise a global temporal interactive multi-head self-attention (**GlobalD**) module which incorporates the relevance and variation of bi-temporal spatial-spectral features, deriving the integrate potential same kind of changes in the local and global range with the combination of GAS. **A new and challenging hyperspectral change detection dataset** is designed for comparison of different approaches. We perform extensive experiments on six real hyperspectral datasets, and our method outperforms the state-of-the-art algorithms with high accuracy and efficiency. 

![image](https://github.com/meiqihu/GlobalMind/blob/main/GlobalMind.png)
# Paper
[GlobalMind: Global multi-head interactive self-attention network for hyperspectral change detection](https://www.sciencedirect.com/science/article/pii/S0924271624001539)

# Data
We perform extensive experiments on six real hyperspectral datasets: 1) **Farmland**,  2) **Hermiston**, 3) **River**,  4) **Bay**,  5) **Barbara**,  6) **GF5B_BI**.

All six hyperspectral binary change detection datasets are provided here [BaiduCloud]:https://pan.baidu.com/s/1-pw4jrd-xp7QHPS067mV7g?pwd=aey7 (code=aey7).

![image](https://github.com/meiqihu/GlobalMind/blob/main/DatasetOfHyperspectralBCD.png)

# Results
The binary change detection results of propose GlobalMIND are available at: [BaiduCloud]: https://pan.baidu.com/s/1bgOzlOTMeu0jCTJkAl42ng?pwd=ny2i (code= ny2i)

# Citation

Please cite our paper if you find it useful for your research.
@article{hu2024globalmind,
  title={GlobalMind: Global multi-head interactive self-attention network for hyperspectral change detection},
  author={Hu, Meiqi and Wu, Chen and Zhang, Liangpei},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={211},
  pages={465--483},
  year={2024},
  publisher={Elsevier}
}
