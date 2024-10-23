package org.example.test;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO; // SVM
import weka.classifiers.lazy.IBk; // KNN
import weka.classifiers.trees.J48; // Decision Tree
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class ImageClassification {

    public static void main(String[] args) {
        // Tải hình ảnh
        BufferedImage flowerImage = loadImage("src/main/java/org/example/img/a1.png"); // Đường dẫn đến ảnh hoa
        BufferedImage animalImage = loadImage("src/main/java/org/example/img/a2.png"); // Đường dẫn đến ảnh động vật

        // Tạo tập dữ liệu
        Instances data = createDataset(flowerImage, animalImage);

        // Định nghĩa các bộ phân loại
        Classifier[] classifiers = {
                new SMO(), // SVM
                new IBk(3), // KNN
                new J48() // Decision Tree
        };

        for (Classifier classifier : classifiers) {
            try {
                // Đo thời gian
                long startTime = System.currentTimeMillis();

                // Huấn luyện bộ phân loại
                classifier.buildClassifier(data);

                // Đo thời gian
                long endTime = System.currentTimeMillis();
                long timeTaken = endTime - startTime;

                // Đánh giá bộ phân loại bằng phương pháp cross-validation
                Evaluation eval = new Evaluation(data);
                eval.crossValidateModel(classifier, data, 2, new java.util.Random(1)); // 2-fold cross-validation

                // Xuất kết quả
                System.out.println("Classifier: " + classifier.getClass().getSimpleName());
                System.out.println("Time taken: " + timeTaken + " ms");
                System.out.println("Accuracy: " + eval.pctCorrect() + "%");
                System.out.println("Precision (class flower): " + eval.precision(0)); // 0 là chỉ số của class flower
                System.out.println("Recall (class flower): " + eval.recall(0)); // 0 là chỉ số của class flower
                System.out.println();

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    // Hàm tải hình ảnh
    private static BufferedImage loadImage(String path) {
        try {
            return ImageIO.read(new File(path));
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    // Hàm tạo tập dữ liệu từ hình ảnh
    private static Instances createDataset(BufferedImage flowerImage, BufferedImage animalImage) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("colorFeature")); // Thuộc tính màu
        attributes.add(new Attribute("textureFeature")); // Thuộc tính kết cấu
        attributes.add(new Attribute("class", List.of("flower", "animal"))); // Lớp

        Instances data = new Instances("ImageData", attributes, 0);
        data.setClass(attributes.get(2));

        // Thêm đặc trưng từ hình ảnh hoa
        double[] flowerValues = new double[data.numAttributes()];
        flowerValues[0] = extractColorFeature(flowerImage); // Trích xuất đặc trưng màu
        flowerValues[1] = extractTextureFeature(flowerImage); // Trích xuất đặc trưng kết cấu
        flowerValues[2] = 0; // Class: flower
        data.add(new DenseInstance(1.0, flowerValues));

        // Thêm đặc trưng từ hình ảnh động vật
        double[] animalValues = new double[data.numAttributes()];
        animalValues[0] = extractColorFeature(animalImage); // Trích xuất đặc trưng màu
        animalValues[1] = extractTextureFeature(animalImage); // Trích xuất đặc trưng kết cấu
        animalValues[2] = 1; // Class: animal
        data.add(new DenseInstance(1.0, animalValues));

        return data;
    }

    // Hàm trích xuất đặc trưng màu từ hình ảnh (đơn giản)
    private static double extractColorFeature(BufferedImage image) {
        // Giả sử chúng ta chỉ lấy giá trị trung bình màu sắc
        int totalPixels = image.getWidth() * image.getHeight();
        long redSum = 0, greenSum = 0, blueSum = 0;

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int rgb = image.getRGB(x, y);
                redSum += (rgb >> 16) & 0xFF;
                greenSum += (rgb >> 8) & 0xFF;
                blueSum += (rgb) & 0xFF;
            }
        }

        double averageColor = (redSum + greenSum + blueSum) / (3.0 * totalPixels);
        return averageColor; // Trả về giá trị trung bình màu sắc
    }

    // Hàm trích xuất đặc trưng kết cấu từ hình ảnh (đơn giản)
    private static double extractTextureFeature(BufferedImage image) {
        // Bạn có thể sử dụng phương pháp trích xuất kết cấu phức tạp hơn ở đây
        // Ở đây ta sử dụng số lượng pixel để đại diện cho đặc trưng kết cấu
        return image.getWidth() * image.getHeight();
    }
}

