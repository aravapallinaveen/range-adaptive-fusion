from PIL import Image
import glob
import os

def main():
    frames = []
    imgs = sorted(glob.glob("demo_frames/*.png"))
    if not imgs:
        print("No frames found!")
        return
        
    for i in imgs:
        # Open and resize to save space in the GIF (scale width to 720, maintain aspect)
        img = Image.open(i)
        w, h = img.size
        new_w = 720
        new_h = int(h * (new_w / w))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        frames.append(img)
        
    # Save as GIF
    out_path = 'demo.gif'
    frames[0].save(out_path, format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=500, loop=0)
               
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
