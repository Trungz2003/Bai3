package org.example;

import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.File;

public class ImageFeatureExtractor {
    public static double[] extractColorHistogram(File imageFile) throws Exception {
        BufferedImage img = ImageIO.read(imageFile);
        int[] histogram = new int[256];

        // Tính toán histogram cho kênh màu đỏ
        for (int x = 0; x < img.getWidth(); x++) {
            for (int y = 0; y < img.getHeight(); y++) {
                int pixel = img.getRGB(x, y);
                int red = (pixel >> 16) & 0xff;
                histogram[red]++;
            }
        }

        // Chuyển đổi histogram thành mảng đặc trưng
        double[] featureVector = new double[histogram.length];
        for (int i = 0; i < histogram.length; i++) {
            featureVector[i] = histogram[i] / (double)(img.getWidth() * img.getHeight());
        }

        return featureVector;
    }
}

