close all; clc; clear all;
disp('Chanel Michaeli ID 208491787');
disp('Noam Atias ID 311394357');

%% Q1.1
cameraman_img = imread('cameraman.tif');
cameraman_img=double(cameraman_img);
cameraman_img = (cameraman_img - min(cameraman_img(:)))/(max(cameraman_img(:))-min(cameraman_img(:)));
figure('Name','Q1.1')
imshow(cameraman_img);
title('Cameramen image normalized');

%% Q1.2.1
thresh = 0.22;
cameraman_prewitt = dip_prewitt_edge(cameraman_img,thresh);
cameraman_prewitt_matlab = edge(cameraman_img,"prewitt",thresh);

figure('Name','Q1.2.1');
subplot(1,2,1);
imshow(cameraman_prewitt);
title('Prewitt edge detection for cameramen image');
subplot(1,2,2);
imshow(cameraman_prewitt_matlab);
title('Prewitt edge detection for cameramen image using matlab function');
sim = ssim(cameraman_prewitt,double(cameraman_prewitt_matlab));
%% Q1.2.2
thresh1 = 0.3;
thresh2 = 0.05;
cameraman_prewitt1 = dip_prewitt_edge(cameraman_img,thresh1);
cameraman_prewitt2 =  dip_prewitt_edge(cameraman_img,thresh2);

figure('Name','Q1.2.2');
subplot(1,2,1);
imshow(cameraman_prewitt1);
title('Prewitt edge detection treshold = 0.3');
subplot(1,2,2);
imshow(cameraman_prewitt2);
title('Prewitt edge detection treshold = 0.05');

%% Q1.3.2&3
[cameraman_canny_default,th] = edge(cameraman_img,'canny');
cameraman_canny_th = edge(cameraman_img,'canny',[0.1 0.6]);
cameraman_canny_sigma = edge(cameraman_img,'canny',[],3);


figure('Name','Q1.3.2');
subplot(1,2,1);
imshow(cameraman_canny_default);
title('Canny edge detection using default parameters');
subplot(1,2,2);
imshow(cameraman_canny_th);
title(['Canny edge detection using parameters: [low,high]=[0.1,0.6], sigma=',num2str(sqrt(2))]);

figure('Name','Q1.3.3')
imshow(cameraman_canny_sigma);
title('Canny edge detection using parameters: sigma=3, default thresholds');

%% Q2.1.1
floor_img = imread('floor.jpg');
floor_img=double(rgb2gray(floor_img));
floor_img = (floor_img - min(floor_img(:)))/(max(floor_img(:))-min(floor_img(:)));
figure('Name','Q2.1.1')
imshow(floor_img);
title('Floor image grayscale normalized');

%% Q2.1.2
BW=edge(floor_img);
figure('Name','Q2.1.2')
imshow(BW);
title('Default edge detection for floor image');

%% Q2.1.3

H1 = dip_hough_lines(BW,1, 1);
H2 = dip_hough_lines(BW,5, 4);
figure('Name','Q2.1.3');
subplot(1,2,1);
imshow(H1,[]);
title('Hough lines detection R_0=1, \theta_0 =1');
subplot(1,2,2);
imshow(H2,[]);
title('Hough lines detection R_0=5, \theta_0 =4');

figure; 
subplot(2,1,1);
imshow(H1,[],'InitialMagnification','fit');
colormap jet; 
colorbar; 
axis normal
title("Hough lines detection R_0=1, \theta_0 =1");
xlabel("\theta"); 
ylabel("r");

subplot(2,1,2);
imshow(H1,[],'InitialMagnification','fit');
colormap jet; 
colorbar; 
axis normal
title("Hough lines detection R_0=5, \theta_0 =4");
xlabel("\theta"); 
ylabel("r");
%% Q1.2.5
% draw peaks
peaks1 = houghpeaks(H1,4,'Threshold',1);
peaks2 = houghpeaks(H2,4,'Threshold',1);
M=size(BW,1);
N=size(BW,2);
R_0=[1,5];
th_0=[1,4];
peaks = {peaks1,peaks2};
for i =1:2
    peaks4 = cell2mat(peaks(i));
    R = -sqrt(M^2+N^2):R_0(i):sqrt(M^2+N^2);
    th = (-90:th_0(i):90);
    figure('Name',['Q1.2.5 peaks of R0=',num2str(R_0(i)),' theta_0=',num2str(th_0(i))])
    imshow(eval(['H',num2str(i)]),[],'XData',th,'YData',R,'InitialMagnification','fit');
    xlabel('\theta'), ylabel('r');axis on, axis normal,hold on;
    plot(th(peaks4(:,2)),R(peaks4(:,1)),'s','color','green');
    hold off
    title(['4 main peaks of Hough matrix R_0=',num2str(R_0(i)),' \theta_0=',num2str(th_0(i))]);
end

%draw lines
for i = 1:2
    peaks4 = cell2mat(peaks(i));
    R = -sqrt(M^2+N^2):R_0(i):sqrt(M^2+N^2);
    th = (-90:th_0(i):90)*(pi/180);
    a = -1 ./ tan(th(peaks4(:,2)));
    x = R(peaks4(:,1)).*cos(th(peaks4(:,2)));
    y = R(peaks4(:,1)).*sin(th(peaks4(:,2)));
    b = y-a.*x;
    n = 1:N;
    figure('Name',['Q1.2.5 lines of R0=',num2str(R_0(i)),' theta_0=',num2str(th_0(i))]);
    imshow(floor_img);
    hold on
    for j = 1:length(a)
        plot(n,a(j)*n+b(j),'LineWidth',2,'color','b');
    end
    hold off
    title(['4 most significant lines R_0=',num2str(R_0(i)),' \theta_0=',num2str(th_0(i))]);

end
%% Q2.2.1
coffee_img = imread('coffee.jpg');
coffee_img=double(rgb2gray(coffee_img));
coffee_img = (coffee_img - min(coffee_img(:)))/(max(coffee_img(:))-min(coffee_img(:)));
figure('Name','Q2.2.1')
imshow(coffee_img);
title('Coffee image grayscale normalized');

%% Q2.2.2
BW_coffee=edge(coffee_img);
figure('Name','Q2.2.2')
imshow(BW_coffee);
title('Default edge detection for coffee image');

%% Q2.2.3
tic;
H1 = dip_hough_circles(BW_coffee,1, 1);
time1 = toc;
H2 = dip_hough_circles(BW_coffee,4,10);
disp(['The run time of calculating hough matrix to find circles is : ', num2str(time1)])
%% Q2.2.4
tic;
H_same = dip_hough_circles(BW_coffee,1,2);
time2 = toc;
disp(['The new run time of calculating hough matrix to find circles is : ', num2str(time2)])
% clac accuracy
e = immse(H1,H_same);
acc = ssim(H1,H_same);

speedup = time1/time2;
%% Q2.2.5
figure('Name','Q2.2.5');
subplot(3,1,1);
imshow(H1(:,:,1),[]);
title('One Silce of Hough 3D with R_0=1,\theta_0=1')
subplot(3,1,2);
imshow(H2(:,:,1),[]);
title('One Silce of Hough 3D with R_0=4,\theta_0=10')
subplot(3,1,3);
imshow(H_same(:,:,1),[]);
title('One Silce of Hough 3D with R_0=1,\theta_0=2')

%% Q2.2.6
peaks3d_1= dip_houghpeaks3d(H1);
peaks3d_2= dip_houghpeaks3d(H2);
peaks3d_same= dip_houghpeaks3d(H_same);

figure('Name','Q2.2.6');
teta = 0:1:360;
b = peaks3d_1(:,1)';
a = peaks3d_1(:,2)';
R = peaks3d_1(:,3)'+79;
subplot(3,1,1);
imshow(coffee_img);
hold on
for i = 1:5
    x = a(i) + R(i)*cos(pi*teta/180);
    y = b(i) + R(i)*sin(pi*teta/180);
    plot(x,y,'LineWidth',2,'color','g');
end
hold off
title('5 most significant circles R_0=1 & theta_0=1');

subplot(3,1,2);
b = peaks3d_2(:,1)';
a = peaks3d_2(:,2)';
R = peaks3d_2(:,3)'+79;
imshow(coffee_img);
hold on
for i = 1:5
    x = a(i) + R(i)*cos(pi*teta/180);
    y = b(i) + R(i)*sin(pi*teta/180);
    plot(x,y,'LineWidth',2,'color','g');
end
hold off
title('5 most significant circles R_0=4 & theta_0=10');

subplot(3,1,3);
b = peaks3d_same(:,1)';
a = peaks3d_same(:,2)';
R = peaks3d_same(:,3)'+79;
imshow(coffee_img);
hold on
for i = 1:5
    x = a(i) + R(i)*cos(pi*teta/180);
    y = b(i) + R(i)*sin(pi*teta/180);
    plot(x,y,'LineWidth',2,'color','g');
end
hold off
title('5 most significant circles R_0=1 & theta_0=2');








function G = dip_prewitt_edge(img,thresh)
G_py = (1/6)*[-1 0 1 ; -1 0 1 ; -1 0 1];
G_px = (1/6)*[1 1 1 ; 0 0 0 ; -1 -1 -1];
Gx = conv2(img,G_px,'same');
Gy = conv2(img,G_py,'same');
G=zeros(size(img));
G(abs(Gx)>thresh)=1;
G(abs(Gy)>thresh)=1;
end


function H = dip_hough_lines(BW,R0, teta0)
M = size(BW,1);
N = size(BW,2);
R = -sqrt(N^2+M^2):R0:sqrt(N^2+M^2);
teta = -90:teta0:90;
H = zeros(length(R),length(teta));
[X,Y] = find(BW);
for i = 1:length(X)
     for t = 1:length(teta)
            r = Y(i)*cos(teta(t)*pi/180)+X(i)*sin(teta(t)*pi/180);
            [~,ind] = min(abs(R-r));
            H(ind,t) = H(ind,t) + 1;
     end
end
end


function H = dip_hough_circles(BW,R0, teta0)
N = size(BW,1);
M = size(BW,2);
Rmin = 80;
Rmax = 100;
R = Rmin:R0:Rmax;
teta= 0:teta0:360;
H = zeros(N,M,length(R));
[X,Y] = find(BW);
for i = 1:length(X)
     for r = 1:length(R)
         for t = 1:length(teta)
            a = round(X(i) - R(r)*cos(teta(t)*pi/180));
            b = round(Y(i) - R(r)*sin(teta(t)*pi/180));
            if (b > 0) & (b <= M) & (a > 0) & (a <= N)
                H(a,b,r) = H(a,b,r) +1;
            end
         end
     end
end
end


function peaks= dip_houghpeaks3d(HoughMatrix)
peaks = zeros(5,3);
HoughMatrix = padarray(HoughMatrix,[6 6 6],'both');
for i= 1:5
    [~ ,idx] = max(HoughMatrix(:));
    [idx1 ,idx2 ,idx3] = ind2sub(size(HoughMatrix),idx);
    peaks(i,:) = [idx1, idx2, idx3];
    HoughMatrix((idx1-6):(idx1+6),(idx2-6):(idx2+6),(idx3-6):(idx3+6)) = 0;
end
peaks = peaks - 6;
end