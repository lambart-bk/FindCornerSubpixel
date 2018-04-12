#include "findcornersubpixel/findcornersubpixel.h"
#include"iostream"
#include"stdlib.h"
#include"iomanip"
using namespace std;
using namespace cv;

void printMat(cv::Mat m,char *info,int flag=1)
{
  cout<<"--------------------"<<endl;
  cout<<info<<":"<<endl;
  if(flag==1)
  {
    for(int i=0;i<m.rows;i++)
    {
      CvPoint2D32f *p_data=m.ptr<CvPoint2D32f>(i);
      for(int j=0;j<m.cols;j++)
      {      
	cout<<"("<<p_data[j].x<<","<<p_data[j].y<<") ";
      }
      cout<<""<<endl;
    }
  }
  else if(flag==2)
  {
    for(int i=0;i<m.rows;i++)
    {
      for(int j=0;j<m.cols;j++)
      {
	cout<<" "<<(int)m.at<unsigned char>(i,j)<<"";
      }
      cout<<endl;
    }
  }
  else if(flag==3)
  {
    for(int i=0;i<m.rows;i++)
    {
      for(int j=0;j<m.cols;j++)
      {
	cout<<" "<<setw(10)<<m.at<float>(i,j)<<" ";
      }
      cout<<endl;
    }
  }
  cout<<"--------------------"<<endl;
}
void compareResult(cv::Mat a,cv::Mat b)
{
  double errX=0,errY=0,x,y;
  CV_Assert(a.cols==b.cols && a.rows==b.rows);
  for(int i=0,k=0;i<a.rows;i++)
    {
      for(int j=0;j<a.cols;j++,k++)
      {
	cout<<k<<":"<<a.at<CvPoint2D32f>(i,j).x-b.at<CvPoint2D32f>(i,j).x<<"\t";
	x=a.at<CvPoint2D32f>(i,j).x-b.at<CvPoint2D32f>(i,j).x;
	errX+=x*x;
	cout<<a.at<CvPoint2D32f>(i,j).y-b.at<CvPoint2D32f>(i,j).y<<""<<endl;
	y=a.at<CvPoint2D32f>(i,j).y-b.at<CvPoint2D32f>(i,j).y;
	errY+=y*y;
      }
      //cout<<endl;
    }
  cout<<"\terrX: "<<errX<<" errY:"<<errY<<endl;
}
void myCornerSubPixOld(cv::Mat *img,CvPoint2D32f *corners,int count,cv::Size win,int max_iter=30,double eps=0.1)
{
  //注：未考虑取样(SubPix)，故与源码结果有所出入
  if(!corners || count<0 || win.width<=0 || win.height<=0 || CV_MAT_TYPE( img->type() ) != CV_8UC1 )
    exit(-1);
  if(count==0)
    return;
  cv::Size imgSize(img->cols ,img->rows),winsize(win.width*2+1,win.height*2+1);
  if(imgSize.width<winsize.width+4 || imgSize.height<winsize.height+4)
    exit(-1);
  
  //calculate mask
  double *maskw=(double*)malloc(winsize.width*sizeof(double));
  double *maskh=(double*)malloc(winsize.height*sizeof(double));
  double coeff=1.f/(win.width*win.width);
  for(int i=-win.width,k=0;i<=win.width;i++,k++)
  {
    maskw[k]=(double)exp(-i*i*coeff);
    //cout<<""<<maskw[k]<<endl;
  }
  if(win.height==win.width)
    maskh=maskw;
  else
  {
    coeff=1.f/(win.height*win.height);
    for(int i=-win.height,k=0;i<win.height;i++,k++)
    {
      maskh[k]=(double)exp(-i*i*coeff);
    }
  }
  
  //calculate derivative (gradient)
  cv::Mat  gradientX(img->rows,img->cols,CV_32FC1),gradientY(img->rows,img->cols,CV_32FC1);
  for(int i=0;i<img->rows;i++)
    for(int j=0;j<img->cols;j++)
    {
      if( i==0 || i== img->rows-1 || j==0 || j==img->cols-1 ) //第一行（列）和最后一行（列）不可导
      {
	gradientX.at<float>(i,j)=0;
	gradientY.at<float>(i,j)=0;
      }
      else
      {
	gradientX.at<float>(i,j)=(int)img->at<unsigned char>(i,j+1)-(int)img->at<unsigned char>(i,j-1);
	gradientY.at<float>(i,j)=(int)img->at<unsigned char>(i+1,j)-(int)img->at<unsigned char>(i-1,j);
	//cout<<(int)gradientX.at<float>(i,j)<<" "<<(int)img->at<unsigned char>(i,j+1)<<","<<(int)img->at<unsigned char>(i,j-1)<<""<<endl;
      }
    }
  
  
  //optimize loop for all corners
  for(int k=0;k<count;k++)
  {
    CvPoint2D32f point=corners[k],c1=point;
    double err=0;
    for(int iter=0;iter<max_iter;iter++)
    {
      double a,b,c,qx,qy;      
      double gx,gy,w,gxx,gxy,gyy;
      int px,py;
      CvPoint2D32f c2;
      a=b=c=qx=qy=0;
      for(int i=-win.height;i<=win.height;i++)
      {
	py=i+(int)(c1.y+0.5);  //px,py为图像坐标，i,j为其相对中心坐标
	if( py<1 || py>imgSize.height-2 ) //原图中第一行和最后一行不考虑，无梯度
	  continue;
	for(int j=-win.width;j<=win.width;j++)
	{
	  px=j+(int)(c1.x+0.5);
	  if( px<1 || px>imgSize.width-2 )  //原图中第一列和最后一列不考虑，无梯度
	    continue;
	  //cout<<px<<","<<py<<" imgsize: "<<imgSize.width<<" img: "<<(int)img->at<unsigned char>(py,px)<<endl;
	  gx=gradientX.at<float>(py,px); //注意：px,py对应行列	
	  gy=gradientY.at<float>(py,px);
	  //cout<<"gx,gy: "<<gx<<","<<gy<<endl;
	  w=maskw[j+win.width]*maskh[i+win.height];
	  //cout<<"w: "<<w<<" "<<j<<","<<i<<endl;
	  gxx=gx*gx*w;
	  gxy=gx*gy*w;
	  gyy=gy*gy*w;
	  
	  a+=gxx;
	  b+=gxy;
	  c+=gyy;
	  
	  //    |gxx gxy| * w * |px|   = |gxx gxy| * w * |qx|  
	  //    |gyx gyy|       |py|     |gyx gyy|       |qy|  
	  //    下面的qx,qy为上式右边值，后面乘以逆可得qx,qy
	  qx+=gxx*j+gxy*i;  //注意：此处应用相对坐标,求得的qx,qy也为相对坐标
	  qy+=gxy*j+gyy*i;
	}
      }
      // |a b|   inversion: | c -b| / det
      // |b c|              |-b  a|
      double det=a*c-b*b;
      double scale=1.f/det;	
      //cout<<det<<","<<scale<<endl;
      if(fabs(det)>DBL_EPSILON*DBL_EPSILON)
      {
	//inverse
	c2.x=c1.x+c*scale*qx-b*scale*qy; //乘以逆得qx,qy,再转为图像绝对坐标
	c2.y=c1.y-b*scale*qx+a*scale*qy;
	//cout<<"qx-"<<c*scale*qx-b*scale*qy<<" ,qy-"<<-b*scale*qx+a*scale*qy<<endl;
      }
      else
      {
	cout<<"warning: the determinant is equal to zero,not inverse! "<<endl;
	cout<<"         stop optimize process in k,iter:"<<" ("<<k<<","<<iter<<")\n"<<endl;
	c2=c1;  
      }
      //squrae of error in comparsion operations
      err=(c1.x-c2.x)*(c1.x-c2.x)+(c1.y-c2.y)*(c1.y-c2.y); 
      c1=c2;	
      //cout<<"c2: "<<c2.x<<","<<c2.y<<endl;
      if(err<eps*eps)
	break;
    }//iter loop
    //if new point is too far from initial,it means poor convergence
    //leave  initial point as the result
    if(fabs(c1.x-point.x)>win.width || fabs(c1.y-point.y)>win.height)
    {
      cout<<"poor convergence in k: "<<k<<endl;
      c1=point;
    }
    corners[k]=c1;
  }//corners loop
  
}

void mygetRectSubPix(cv::Mat src,cv::Mat &roi,cv::Size &roi_sz, CvPoint2D32f point)
{

  int xi,yi;
  float x=point.x,y=point.y,dx,dy,p00,p01,p10,p11;
  xi=cvFloor(x);
  yi=cvFloor(y);
  dx=x-xi;
  dy=y-yi;
  //如果窗口越界，缩小窗口
  for(int i=roi_sz.height,j=roi_sz.width;i>0,j>0;i--,j--)
  {
    if(xi-j<0 || xi+j+1>=src.cols || yi-i<0 || yi+i+1>=src.rows)
      continue;
    else
    {
      roi_sz.height=i;
      roi_sz.width=j;
      break;
    }
  }
  cv::Mat temp(roi_sz.height*2+1,roi_sz.width*2+1,CV_32FC1);//  roi-RectSubpix
  for(int i=-roi_sz.height,ii=0;i<=roi_sz.height;i++,ii++)
    for(int j=-roi_sz.width,jj=0;j<=roi_sz.width;j++,jj++)
    {
      p00=(float)src.at<unsigned char>(yi+i,xi+j);
      p01=(float)src.at<unsigned char>(yi+i,xi+j+1);
      p10=(float)src.at<unsigned char>(yi+i+1,xi+j);
      p11=(float)src.at<unsigned char>(yi+i+1,xi+j+1);
      //cout<<"p00,p01,p10,p11: "<<p00<<","<<p01<<","<<p10<<","<<p11<<endl;
      //cout<<"--: "<<( (1-dx)*p00+dx*p10) * (1-dy) + ( (1-dx)*p01+dx*p11) * dy<<endl;
      temp.at<float>(ii,jj)=( (1-dx)*p00+dx*p10) * (1-dy) + ( (1-dx)*p01+dx*p11) * dy;
    }
  temp.copyTo(roi);
  //printMat(temp,"temp",3);
  
}

void myCornerSubPix(cv::Mat *img,CvPoint2D32f *corners,int count,cv::Size win,int max_iter=30,double eps=0.1)
{
  if(!corners || count<0 || win.width<=0 || win.height<=0 || CV_MAT_TYPE( img->type() ) != CV_8UC1 )
    exit(-1);
  if(count==0)
    return;
  cv::Size imgSize(img->cols ,img->rows),winsize(win.width*2+1,win.height*2+1);
  if(imgSize.width<winsize.width+4 || imgSize.height<winsize.height+4)
    exit(-1);
  
  //calculate mask
  double *maskw=(double*)malloc(winsize.width*sizeof(double));
  double *maskh=(double*)malloc(winsize.height*sizeof(double));
  double coeff=1.f/(win.width*win.width);
  for(int i=-win.width,k=0;i<=win.width;i++,k++)
  {
    maskw[k]=(double)exp(-i*i*coeff);
    //cout<<""<<maskw[k]<<endl;
  }
  if(win.height==win.width)
    maskh=maskw;
  else
  {
    coeff=1.f/(win.height*win.height);
    for(int i=-win.height,k=0;i<win.height;i++,k++)
    {
      maskh[k]=(double)exp(-i*i*coeff);
    }
  } 
  
  //optimize loop for all corners
  for(int k=0;k<count;k++)
  {
    CvPoint2D32f point=corners[k],c1=point;
    double err=0;
    cv::Size win_local;
    
    for(int iter=0;iter<max_iter;iter++)
    {
      
      //getRectSubpix / roi
      cv::Mat roi;
      cv::Size roi_sz(win.width+1,win.height+1);
      mygetRectSubPix(*img,roi,roi_sz,c1);
      
      win_local.width=roi_sz.width-1;
      win_local.height=roi_sz.height-1;
      
      //calculate derivative (gradient)
      cv::Mat gradientX,gradientY;
      cv::Mat  gradientXt(roi_sz.height*2+1,roi_sz.width*2+1,CV_32FC1),gradientYt(roi_sz.height*2+1,roi_sz.width*2+1,CV_32FC1);
      CV_Assert(roi.rows==roi_sz.height*2+1 && roi.cols==roi_sz.width*2+1);
      for(int i=0;i<roi.rows;i++)
	for(int j=0;j<roi.cols;j++)
	{
	  if( i==0 || i== roi.rows-1 || j==0 || j==roi.cols-1 ) //第一行（列）和最后一行（列）不可导
	  {
	    gradientXt.at<float>(i,j)=0;
	    gradientYt.at<float>(i,j)=0;
	  }
	  else
	  {
	    gradientXt.at<float>(i,j)=roi.at<float>(i,j+1)-roi.at<float>(i,j-1);
	    gradientYt.at<float>(i,j)=roi.at<float>(i+1,j)-roi.at<float>(i-1,j);
	  }
	}
      gradientXt.rowRange(1,roi.rows-1).colRange(1,roi.cols-1).copyTo(gradientX);
      gradientYt.rowRange(1,roi.rows-1).colRange(1,roi.cols-1).copyTo(gradientY);
      
      //printMat(roi,"roi",3);
      //printMat(gradientX,"gx",3);
      
      
      double a,b,c,qx,qy;      
      double gx,gy,w,gxx,gxy,gyy;
      int px,py;
      CvPoint2D32f c2;
      a=b=c=qx=qy=0;
      for(int i=-win_local.height;i<=win_local.height;i++)
      {
	py=i+(int)(c1.y+0.5);  //px,py为图像坐标，i,j为其相对中心坐标
	CV_Assert(py>=0 && py<img->rows);
	for(int j=-win_local.width;j<=win_local.width;j++)
	{
	  px=j+(int)(c1.x+0.5);
	  CV_Assert(px>=0 && px<img->cols);
	  //cout<<px<<","<<py<<" imgsize: "<<imgSize.width<<" img: "<<(int)img->at<unsigned char>(py,px)<<endl;
	  gx=gradientX.at<float>(i+win_local.height,j+win_local.width); 	
	  gy=gradientY.at<float>(i+win_local.height,j+win_local.width);
	  //cout<<"gx,gy: "<<gx<<","<<gy<<endl;
	  w=maskw[j+win_local.width]*maskh[i+win_local.height];
	  //cout<<"w: "<<w<<" "<<j<<","<<i<<endl;
	  gxx=gx*gx*w;
	  gxy=gx*gy*w;
	  gyy=gy*gy*w;
	  
	  a+=gxx;
	  b+=gxy;
	  c+=gyy;
	  
	  //    |gxx gxy| * w * |px|   = |gxx gxy| * w * |qx|  
	  //    |gyx gyy|       |py|     |gyx gyy|       |qy|  
	  //    下面的qx,qy为上式右边值，后面乘以逆可得qx,qy
	  qx+=gxx*j+gxy*i;  //注意：此处应用相对坐标,求得的qx,qy也为相对坐标
	  qy+=gxy*j+gyy*i;
	}
      }
      // |a b|   inversion: | c -b| / det
      // |b c|              |-b  a|
      double det=a*c-b*b;
      double scale=1.f/det;	
      //cout<<det<<","<<scale<<endl;
      if(fabs(det)>DBL_EPSILON*DBL_EPSILON)
      {
	//inverse
	c2.x=c1.x+c*scale*qx-b*scale*qy; //乘以逆得qx,qy,再转为图像绝对坐标
	c2.y=c1.y-b*scale*qx+a*scale*qy;
	//cout<<"qx-"<<c*scale*qx-b*scale*qy<<" ,qy-"<<-b*scale*qx+a*scale*qy<<endl;
      }
      else
      {
	cout<<"warning: the determinant is equal to zero,not inverse! "<<endl;
	cout<<"         stop optimize process in k,iter:"<<" ("<<k<<","<<iter<<")\n"<<endl;
	c2=c1;  
      }
      //squrae of error in comparsion operations
      err=(c1.x-c2.x)*(c1.x-c2.x)+(c1.y-c2.y)*(c1.y-c2.y); 
      c1=c2;	
      //cout<<"c2: "<<c2.x<<","<<c2.y<<endl;
      if(err<eps*eps)
	break;
    }//iter loop
    //if new point is too far from initial,it means poor convergence
    //leave  initial point as the result
    if(fabs(c1.x-point.x)>win_local.width || fabs(c1.y-point.y)>win_local.height)
    {
      cout<<"poor convergence in k: "<<k<<endl;
      c1=point;
    }
    corners[k]=c1;
  }//corners loop
  
  
}
int main(int argc,char**argv)
{

  if(argc<2)
  {
    cout<<"usage: "<<argv[0]<<" path_to_pattern"<<endl;
    return -1;
  }
  cv::Mat img=cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
  cv::namedWindow("img");
  //printMat(img,"img_raw",2);
  cv::Mat outimg=img.clone(),corners,corners1,corners2;
  cv::Size patternSize(8,6);
  bool patternFound=cv::findChessboardCorners(img,cv::Size(8,6),corners);
  cout<<"corners.checkVector(2)= "<<corners.checkVector(2)<<endl;
  cout<<"corners.total()= "<<corners.total()<<"\tcorners.channels= "<<corners.channels()<<endl;
  cout<<"patternFound: "<<patternFound<<" row:"<<corners.rows<<" cols:"<<corners.cols<<endl;
  //printMat(corners,"corner_raw");
  //cv::drawChessboardCorners(outimg,patternSize,corners,patternFound);
  //cv::imshow("img",outimg);
  //cv::waitKey(0);
  corners1=corners.clone();
  cv::cornerSubPix(outimg,corners1,cv::Size(11,11),cv::Size(-1,-1),
		   cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,0.1));
  //printMat(corners1,"corner1");
  
  corners2=corners.clone();
  //myCornerSubPixOld(&outimg,(CvPoint2D32f*)corners2.data,1/*corners2.checkVector(2)*/,cv::Size(1,1),10,0.1);
  //cout<<corners2.at<CvPoint2D32f>(0,0).x<<","<<corners2.at<CvPoint2D32f>(0,0).y<<endl;
  //myCornerSubPixOld(&outimg,(CvPoint2D32f*)corners2.data,corners2.checkVector(2),cv::Size(11,11),30,0.1);//errX: 3.64577 errY:3.99414
  
  //myCornerSubPix(&outimg,(CvPoint2D32f*)corners2.data,1/*corners2.checkVector(2)*/,cv::Size(1,1),10,0.1);
  myCornerSubPix(&outimg,(CvPoint2D32f*)corners2.data,corners2.checkVector(2),cv::Size(11,11),30,0.1);//errX: 6.20713 errY:6.07217   why 未考虑取样的居然与opencv实现的结果更接近？哪里错了？             
  
  //printMat(corners2,"corner2");
  
  //compareResult(corners1,corners2);
  
  cv::destroyAllWindows();
  return 0;
}
