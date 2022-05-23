package com.xincheck.dl.imagecheck.demo;

import com.xincheck.dl.imagecheck.model.VisionTransformer;
import com.xincheck.dl.imagecheck.model.TransformerFactory;
import com.xincheck.dl.imagecheck.exception.UnsupportedModelType;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws UnsupportedModelType {
        VisionTransformer.setModelPath("src/main/resources/weights/");
        VisionTransformer model = null;

        try {
            BufferedImage srcImg = ImageIO.read(new File("./demo/1.png"));
            BufferedImage refImg = ImageIO.read(new File("./demo/2.png"));

            model = TransformerFactory.getVisionTransformer("vit");
            System.out.println(model.eval(srcImg, refImg));

            model = TransformerFactory.getVisionTransformer("swin");
            System.out.println(model.eval(srcImg, refImg));

            model = TransformerFactory.getVisionTransformer("cmt");
            System.out.println(model.eval(srcImg, refImg));

            model = TransformerFactory.getVisionTransformer("deit");
            System.out.println(model.eval(srcImg, refImg));
        } catch (IOException e) {
            System.err.println("读取图片时发生错误，请检查图片路径是否正确、图片是否损坏等");
        }
    }

}
