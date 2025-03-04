import py4j.GatewayServer;
import java.util.ArrayList;

public class Distance {
    byte[][][] testImages;
    byte[][][] trainingImages;

    byte[][][] transformedTest;
    byte[][][] transformedTraining;

    byte[] testReturn;
    byte[] trainingReturn;

    final int IMG_SIZE = 784;
    final int IMG_WIDTH = 28;

    public void clear() {
        System.out.println("CLEARING ALL IMAGE DATA");
        testImages = null;
        trainingImages = null;
        transformedTest = null;
        transformedTraining = null;
        testReturn = null;
        trainingReturn = null;
    }

    public void debug(int i) {
        if (testImages != null) {
            printImg(testImages[i]);
        }
        if (transformedTest != null) {
            printImg(transformedTest[i]);
        }
    }

    public void loadTestData(byte[] pre_imgs) {
        int numImgs = pre_imgs.length / IMG_SIZE;
        System.out.println("Loading " + numImgs + " test images");
        testImages = new byte[numImgs][IMG_WIDTH][IMG_WIDTH];
        for (int i=0; i<numImgs; i++) {
            for (int row=0; row<IMG_WIDTH; row++) {
                for (int col=0; col<IMG_WIDTH; col++) {
                    testImages[i][row][col] = pre_imgs[i * IMG_SIZE + row * IMG_WIDTH + col];
                }
            }
        }
        System.out.println("Loaded test images");
    }
    public void loadTrainingData(byte[] pre_train_imgs) {
        int numImgs = pre_train_imgs.length / IMG_SIZE;
        System.out.println("Loading " + numImgs + " training images");
        trainingImages = new byte[numImgs][IMG_WIDTH][IMG_WIDTH];
        for (int i=0; i<numImgs; i++) {
            for (int row=0; row<IMG_WIDTH; row++) {
                for (int col=0; col<IMG_WIDTH; col++) {
                    trainingImages[i][row][col] = pre_train_imgs[i * IMG_SIZE + row * IMG_WIDTH + col];
                }
            }
        }
        System.out.println("Loaded training images");
    }

    public byte[] distance_transform() {
        if (testImages == null) {
            return null;
        }
        if (testReturn != null) {
            //Cached results
            return testReturn;
        }
        //TODO: Can cut space in half by not using the array
        transformedTest = new byte[testImages.length][28][28];
        testReturn = new byte[testImages.length * IMG_SIZE];

        Thread[] threads = new Thread[testImages.length];
        for (int i=0; i<testImages.length; i++) {
            final int idx = i;

            threads[idx] = new Thread(new Runnable() {
                @Override
                public void run() {
                    transformedTest[idx] = Distance.transform(testImages[idx]);
                    for (int row=0; row<IMG_WIDTH; row++) {
                        for (int col=0; col<IMG_WIDTH; col++) {
                            testReturn[idx * IMG_SIZE + row * IMG_WIDTH + col] = transformedTest[idx][row][col];
                        }
                    }
                }
            });
            threads[i].start();
        }

        try {
            for (Thread thread : threads) {
                thread.join();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("Transformed test images");
        return testReturn;
    }
    public byte[] distance_transform_training() {
        if (trainingImages == null) {
            return null;
        }
        if (trainingReturn != null) {
            //Cached results
            return trainingReturn;
        }
        //TODO: Can cut space in half by not using the array
        transformedTraining = new byte[trainingImages.length][28][28];
        trainingReturn = new byte[trainingImages.length * IMG_SIZE];

        Thread[] threads = new Thread[trainingImages.length];
        for (int i=0; i<trainingImages.length; i++) {
            final int idx = i;

            threads[idx] = new Thread(new Runnable() {
                @Override
                public void run() {
                    transformedTraining[idx] = Distance.transform(trainingImages[idx]);
                    for (int row=0; row<IMG_WIDTH; row++) {
                        for (int col=0; col<IMG_WIDTH; col++) {
                            trainingReturn[idx * IMG_SIZE + row * IMG_WIDTH + col] = transformedTraining[idx][row][col];
                        }
                    }
                }
            });
            threads[i].start();
        }

        try {
            for (Thread thread : threads) {
                thread.join();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        
        System.out.println("Transformed training images");
        return trainingReturn;
    }

    public static byte[][] transform(byte[][] image) {
        //TODO: switch to byte[] return type
        ArrayList<IntTuple> active = new ArrayList<>();
        byte[][] transformedImage = new byte[28][28];
        
        //Parallelizable
        for (byte row = 0; row < 28; row++) {
            for (byte col = 0; col < 28; col++) {
                if (image[row][col] == 1) {
                    active.add(new IntTuple(row, col));
                }
            }
        }

        byte d = (byte) 0xff;
        byte dt = 0;
        
        //Parallelizable
        for (byte row = 0; row < 28; row++) {
            for (byte col = 0; col < 28; col++) {
                if (image[row][col] == 1) {
                    //Skip active points
                    continue;
                }
                d = (byte) 0xff;

                for (IntTuple point : active) {
                    dt = point.distance(row, col);
                    d = (Byte.compareUnsigned(d, dt) > 0 ? dt : d);
                }
                transformedImage[row][col] = d;
            }
        }
        return transformedImage;
    }

    public static void printImg(byte[][] img) {
        for (byte[] row : img) {
            for (byte pixel : row) {
                System.out.print((int) pixel);
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        Distance app = new Distance();
        // app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(app);
        server.start();

        System.out.println("Running!");
    }
}

class IntTuple {
    public byte first;
    public byte second;
    public IntTuple(byte x, byte y) {
        first = x;
        second = y;
    }
    public IntTuple() {
        first = 0;
        second = 0;
    }
    public byte distance(byte x, byte y) {
        return (byte) Math.sqrt(Math.pow(first - x, 2) + Math.pow(second - y, 2));
    }
}