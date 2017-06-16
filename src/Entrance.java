import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import pers.season.vml.statistics.shape.ShapeInstance;
import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.statistics.shape.ShapeModelTrain;
import pers.season.vml.statistics.texture.TextureModel;
import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;

public class Entrance {
	static {
		System.loadLibrary("lib/opencv_java320_x64");
	}

	public final static void main(String[] args) {
		
		// train and init Shape-Model
		MuctData.init("muct/jpg", "muct/muct76-opencv.csv", new int[]{});
		ShapeModelTrain.train("models/shape/", 0.90, false);
		ShapeModel sm = ShapeModel.load("models/shape/", "V", "Z_e");


		// create surrounding anchor points and Delaunay
		int testImg = 1;
		Mat srcpic = MuctData.getGrayJpg(testImg);
		Mat srcpts = MuctData.getPtsMat(testImg);
		Mat srcptsX = new Mat(srcpts.rows() / 2, srcpts.cols(), srcpts.type());
		Mat srcptsY = new Mat(srcpts.rows() / 2, srcpts.cols(), srcpts.type());
		for (int i = 0; i < srcpts.rows() / 2; i++) {
			srcpts.row(i * 2).copyTo(srcptsX.row(i));
			srcpts.row(i * 2 + 1).copyTo(srcptsY.row(i));
		}
		MinMaxLocResult xmm = Core.minMaxLoc(srcptsX);
		MinMaxLocResult ymm = Core.minMaxLoc(srcptsY);
		
		double effectArea = 0.5;
		double xgap = (xmm.maxVal - xmm.minVal) * effectArea;
		double ygap = (ymm.maxVal - ymm.minVal) * effectArea;
		xmm.minVal = xmm.minVal - xgap < 0 ? 0 : xmm.minVal - xgap;
		xmm.maxVal = xmm.maxVal + xgap > srcpic.width() ? srcpic.width() : xmm.maxVal + xgap;
		ymm.minVal = ymm.minVal - ygap < 0 ? 0 : ymm.minVal - ygap;
		ymm.maxVal = ymm.maxVal + ygap > srcpic.height() ? srcpic.height() : ymm.maxVal + ygap;
		Mat adiPts = new Mat();
		int warpDensity = 3;
		for (int y = 0; y <= warpDensity; y++) {
			for (int x = 0; x <= warpDensity; x++) {
				if (x == 0 || y == 0 || x == warpDensity || y == warpDensity) {
					Mat t = new Mat(2, 1, CvType.CV_32F);
					t.put(0, 0, xmm.minVal + (xmm.maxVal - xmm.minVal) * ((double) x / warpDensity),
							ymm.minVal + (ymm.maxVal - ymm.minVal) * ((double) y / warpDensity));
					adiPts.push_back(t);
				}
			}
		}
		srcpts.push_back(adiPts);
		JFrame winDelaunay = new JFrame();
		int[][] delaunay = TextureModel.createDelaunay(new Rect(0, 0, srcpic.width() + 1, srcpic.height() + 1), srcpts);
		ImUtils.showDelaunay(winDelaunay, srcpts, delaunay, srcpic.width(), srcpic.height());
		
		
		// twist according to shape model
		double stepVar = 1.5e-2;
		ShapeInstance shape = new ShapeInstance(sm);
		shape.setFromPts(MuctData.getPtsMat(testImg));
		JFrame win = new JFrame();
		for (int feature = 0; feature < sm.Z_SIZE; feature++) {
			win.setTitle("Feature = " + feature);
			double[] seq = new double[] { 0, 3, -3, 0 };
			for (int s = 0; s < seq.length - 1; s++) {
				for (double i = seq[s]; Math.abs(i - seq[s + 1]) > 0.001; i += 0.5 * Math.signum(seq[s + 1] - seq[s])) {
					Mat z = shape.Z.clone();
					z.put(feature, 0, z.get(feature, 0)[0] + stepVar * i * shape.getScale());
					Mat dstpts = sm.getXfromZ(z);
					dstpts.push_back(adiPts);

					Mat dstpic = srcpic.clone();

					TextureModel.AffineTexture(srcpic, srcpts, dstpic, dstpts, delaunay);
					// ImUtils.showDelaunay(winDelaunay, dstpts, delaunay, dstpic.width(), dstpic.height());
					ImUtils.imshow(win, dstpic, 1);
					System.gc();

				}
			}

		}

	}
}
