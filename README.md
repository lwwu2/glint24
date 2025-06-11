# Position-Normal Manifold for Efficient Glint Rendering on High-Resolution Normal Maps
This repo contains the implementation of the SIGGRAPH 2025 paper: **Position-Normal Manifold for Efficient Glint Rendering on High-Resolution Normal Maps**.

- `src/bsdfs/` contains the Mitsuba 0.6 implementation of specular glint BRDF and diffuse BRDF.

- `convert.py` converts input normal map to the cluster and bounding box hierarchy.

- `convert_G.py` converts input normal map into GGX roughness and tangent frame for fast shadow-masking approximation.

### [Paper](https://arxiv.org/abs/2505.08985) | [Citation](#citation)



## Usage

BSDF plugin in Mitsuba 0.6's XML format:

`````` xml
<bsdf type="glintconductor">
            <string name="micro_normal" value=$normalmap_folder/>
            <float name="sigma" value=$footprint_scale/>
            <float name="thresh" value="0.0004"/> <!--cluster error threshold-->
            <boolean name="fix_sigma" value="false"/> <!--use ray differential for footprint estimation-->
            <float name="texture_scale" value="16.0"/> <!--scale of the texture-->
            <integer name="res" value="1024"/> <!--normal map resolution-->
            <string name="filter" value="gaussian"/> <!--footpint kernel type (gaussian|disk|box)-->
</bsdf>

<bsdf type="glintdiffuse">
            <rgb value="0.8 0.8 0.8" name="reflectance"/>
            <string name="micro_normal" value=$normalmap_folder/>
            <float name="sigma" value=$footprint_scale/>
            <boolean name="fix_sigma" value="false"/> 
            <float name="texture_scale" value="16.0"/>
            <integer name="res" value="1024"/>
</bsdf>
``````

Normal map to cluster and bounding box hierarchy:

```shell
python convert.py --input <normal-or-height-map-file> --output <output-folder>
```

An example is provided in `demo/` that reproduces Fig. 11 of the paper:
1. Download [Mitsuba 0.6](https://github.com/mitsuba-renderer/mitsuba). Copy `src/bsdfs/` to the Mitsuba 0.6 folder and compile according to the [documentation](https://mitsuba-renderer.org/docs.html).
2. In this folder, `cd demo` and run `sh demo.sh`.
3. The rendering results will be saved as `scene1_isotropic.exr`, `scene1_brush.exr`, `scene1_scratched.exr`.



## Citation

``````
@inproceedings{wu2025position,
	author = {Liwen Wu and Fujun Luan and Miloš Hašan and Ravi Ramamoorthi},
	title = {Position-Normal Manifold for Efficient Glint Rendering on High-Resolution Normal Map},
	booktitle = {SIGGRAPH},
	year = {2025}
}
``````

