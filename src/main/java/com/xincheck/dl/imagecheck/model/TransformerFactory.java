package com.xincheck.dl.imagecheck.model;

//import com.xincheck.core.exception.UnsupportedModelType;
import com.xincheck.dl.imagecheck.exception.UnsupportedModelType;


public class TransformerFactory {
    public static VisionTransformer getVisionTransformer(String transformer) throws UnsupportedModelType {
        switch (transformer) {
            case "vit":
                return ViT.getVit();
            case "swin":
                return Swin.getSwinTransformer();
            case "cmt":
                return Conformer.getConformer();
            case "deit":
                return DeiT.getDeiT();
            default:
                throw new UnsupportedModelType();
        }
    }
}