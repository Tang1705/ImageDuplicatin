package com.xincheck.dl.imagecheck.model;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.translate.TranslateException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public abstract class VisionTransformer {
    protected static String modelPath;
    protected Predictor<Image, Classifications> predictor;

    protected Classifications inference(BufferedImage image) {
        Classifications prediction = null;
        try {
            ByteArrayOutputStream os = new ByteArrayOutputStream();
            ImageIO.write(image, "png", os);
            InputStream is = new ByteArrayInputStream(os.toByteArray());

            Image img = ImageFactory.getInstance().fromInputStream(is);
            img.getWrappedImage();

            prediction = this.predictor.predict(img);
        } catch (IOException | TranslateException e) {
            e.printStackTrace();
        }
        return prediction;
    }

    public double[] encode(BufferedImage image) {
        List<Classifications.Classification> img = inference(image).items();
        int size = img.size();
        double[] classToken = new double[size];
        for (int i = 0; i < size; i++) {
            classToken[i] = img.get(i).getProbability();
        }
        return classToken;
    }

    public static void setModelPath(String path) {
        modelPath = path;
    }

    public String getModelPath() {
        return modelPath;
    }

    public double similarity(double[] src, double[] ref) {
        int size = src.length;
        double simVal = 0;

        double num = 0;
        double den = 1;
        double powSrcSum = 0;
        double powRefSum = 0;
        for (int i = 0; i < size; i++) {
            num = num + src[i] * ref[i];
            powSrcSum = powSrcSum + Math.pow(src[i], 2);
            powRefSum = powRefSum + Math.pow(ref[i], 2);
        }
        double sqrtSrc = Math.sqrt(powSrcSum);
        double sqrtRef = Math.sqrt(powRefSum);
        den = sqrtSrc * sqrtRef;

        simVal = num / den;

        return simVal;
    }

    public abstract double eval(BufferedImage srcImg, BufferedImage refImg);

    public abstract double eval(double[] srcImg, double[] refImg);

}
