package com.xincheck.dl.imagecheck.model;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.training.util.DownloadUtils;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class Conformer extends VisionTransformer {

    private static class ConformerHolder {
        private static final Conformer INSTANCE = new Conformer();
    }

    private Conformer() {
        try {
            if (!modelPath.endsWith("/")) {
                modelPath += "/";
            }

            Path modelDir = Paths.get(modelPath);
            Model model = Model.newInstance("Conformer_PyTorch");

            DownloadUtils.download("https://5618.oss-cn-beijing.aliyuncs.com/conformer_model.pt", modelPath + "conformer_model.pt", new ProgressBar());
            DownloadUtils.download("https://5618.oss-cn-beijing.aliyuncs.com/conformer_class.txt", modelPath + "conformer_class.txt", new ProgressBar());

            model.load(modelDir, "conformer_model.pt");

            Pipeline pipeline = new Pipeline();
            pipeline.add(new Resize(256))
                    .add(new CenterCrop(224, 224))
                    .add(new ToTensor())
                    .add(new Normalize(
                            new float[]{
                                    0.485f, 0.456f, 0.406f},
                            new float[]{
                                    0.229f, 0.224f, 0.225f}));

            Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
                    .setPipeline(pipeline)
                    .optSynsetArtifactName("conformer_class.txt")
                    .build();

            this.predictor = model.newPredictor(translator);
        } catch (IOException | MalformedModelException e) {
            e.printStackTrace();
        } catch (NullPointerException e) {
            System.err.println("请先设置模型保存路径");
            System.exit(1);
        }

    }

    public static Conformer getConformer() {
        return ConformerHolder.INSTANCE;
    }

    public double eval(BufferedImage srcImg, BufferedImage refImg) {
        List<Classifications.Classification> src = inference(srcImg).items();
        List<Classifications.Classification> ref = inference(refImg).items();

        double[] srcClassToken = new double[2112];
        double[] refClassToken = new double[2112];
        for (int i = 0; i < 2112; i++) {
            srcClassToken[i] = src.get(i).getProbability();
            refClassToken[i] = ref.get(i).getProbability();
        }

        return similarity(srcClassToken, refClassToken);
    }

    public double eval(double[] srcImg, double[] refImg) {
        return similarity(srcImg, refImg);
    }
}
