package com.xincheck.imagecheck.demo;

import com.xincheck.imagecheck.model.architecture.VisionTransformer;
import com.xincheck.imagecheck.model.transformer.TransformerFactory;
import com.xincheck.imagecheck.entity.VImage;
import com.xincheck.imagecheck.exception.UnsupportedModelType;

public class Main {
    public static void main(String[] args) throws UnsupportedModelType {
        VisionTransformer.setModelPath("src/main/resources/weights/");
        VisionTransformer model = null;

        VImage srcImg = VImage.load("./demo/1.jpg");
        VImage refImg = VImage.load("./demo/3.jpg");

        model = TransformerFactory.getVisionTransformer("vit");
        System.out.println(model.eval(srcImg, refImg));

        model = TransformerFactory.getVisionTransformer("swin");
        System.out.println(model.eval(srcImg, refImg));

        model = TransformerFactory.getVisionTransformer("cmt");
        System.out.println(model.eval(srcImg, refImg));

    }


}
