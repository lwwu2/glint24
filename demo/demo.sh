# construct hierarchies
python ../convert.py --input isotropic.exr --output ./
python ../convert.py --input brush.exr --output ./ --normal
python ../convert.py --input scratched.exr --output ./


# render scene
mitsuba scene1.xml -D t=0.004 -D n=isotropic -o scene1_isotropic.exr
mitsuba scene1.xml -D t=0.004 -D n=brush -o scene1_brush.exr
mitsuba scene1.xml -D t=0.0004 -D n=scratched -o scene1_scratched.exr