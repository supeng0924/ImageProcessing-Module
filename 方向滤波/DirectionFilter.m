function [  ] = DirectionFilter( input_args )
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
%%
%读入图像
image=imread('Lena.jpg');
%初始化全为0，与image同尺寸的图像
deal=zeros(size(image));
[rows,cols]=size(image);
%%
%输入变量区域
sigma=10;
theta=0;
%生成0:30:150 总共6个方向的滤波后结果图对比
%以图像 x y为正方向 theta是与x轴正方向夹角  theta范围是0-179
theta_range=0:30:150;
for theta_i=1:numel(theta_range)
    theta=theta_range(theta_i);
%%
%计算权值矩阵
%权值长度
weight_length=4*sigma+1;
%权值长度一半
half_length=floor(weight_length/2);
%权值矩阵初始化
weight=zeros(1,weight_length);
%斜率
k=tan(theta/180*pi);
%生成t变量
t_val=(1:weight_length)-half_length-1;
%计算实际dert_x dert_y 变化量
if theta==0
    dert_y=zeros(size(t_val));
    dert_x=-half_length:half_length;
elseif theta==90
    dert_y=-half_length:half_length;
    dert_x=zeros(size(t_val));
elseif (theta>0&&theta<45)||(theta>90&&theta<135)
    dert_y=t_val
    dert_x=t_val/k;
    dert_x=round(dert_x)
%     plot(act_x,act_y);axis equal
else
    dert_x=t_val
    dert_y=t_val*k;
    dert_y=round(dert_y)
%     plot(act_x,act_y);axis equal
end

distance=dert_y.*dert_y+dert_x.*dert_x;
weight=exp(-distance./(2*sigma*sigma));
%%
%像素点处理
for row=1:rows
    for col=1:cols
        %计算每一个像素点的值
        
        %累计像素点值
        pixel_total=0;
        %累计概率
        p_total=0;
        
        for i=1:weight_length
            %计算出实际图像中 x y 坐标
            act_x=dert_x(i)+col;
            act_y=dert_y(i)+row;
            %对于超出图像范围内的像素点 舍去
            if act_x>0&&act_x<=cols&&act_y>0&&act_y<=rows
                pixel_total=double(pixel_total)+double(image(act_y,act_x))*weight(i);
                p_total=p_total+weight(i);
            end
        end
        %最终像素点的像素值
        value=pixel_total/p_total;
        deal(row,col)=uint8(value);    
    end
end
%%
%输出图像
subplot(2,3,theta_i);imshow(deal,[]);title(num2str(theta));
end
end

