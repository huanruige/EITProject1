function gray = colorbar_to_image()
    % 如需自动加载 EIDORS 环境，取消下面注释
    % run('D:\Program Files\eidors-v3.12-ng\eidors\startup.m')

    %% ==================== 1. 建模部分 ====================
    n_elecs_per_plane_1 = 16;
    th = linspace(0,360, n_elecs_per_plane_1+1)'; 
    th(1) = [];
    els = (th+67.5)*[1,0]; % [radius (clockwise), z=0]

    % 电极尺寸（按原代码）
    elec_sz = 0.025; % 按照实际的电极与半径的比例

    % 小桶直径 20 cm，网格密度 0.6（原始建模）
    fmdl = ng_mk_cyl_models([0,20,0.6], els, [elec_sz,0,0.5]);

    % 电极接触阻抗
    for i = 1:n_elecs_per_plane_1
        fmdl.electrode(i).z_contact = 0.01;
    end

    % 设置电压和电流刺激模式（原代码）
    fmdl.stimulation = mk_stim_patterns(16,1,[0,1],[0,1], ...
                                        {'no_meas_current','rotate_meas'},1);

    imdl = mk_common_model('c2C',16);
    imdl.fwd_model = fmdl;

    % 做一次示例前向+反演，用于得到一个带 GN 反演信息的 img_gn 模板
    img_1 = mk_image(fmdl, 1);
    select_fcn1 = inline('(x-80).^2+(y-20).^2<20^2','x','y','z');
    img_3 = img_1;
    img_3.elem_data = 1 + elem_select(img_3.fwd_model, select_fcn1);

    vi = fwd_solve(img_3);
    vh = fwd_solve(img_1);

    imdlgn = imdl;
    imdlgn.solve = @inv_solve_diff_GN_one_step;
    img_gn = inv_solve(imdlgn, vi, vh);   % 之后用它作为模板，只替换 elem_data

    %% ==================== 2. 准备文件夹 ====================
    % 成像“预测电导率”的路径
    root_dir = 'E:\GHR\刘金行代码整理\EITProject\predict_ddl\two_objects_dif\VGG16\Best_dice_20';
    % root_dir = 'E:\GHR\刘金行代码整理\EITProject\predict_ddl\two_objects_dif\ECA_ResNet101\Best_dice_30';
    % 原图电导率路径
    % root_dir = 'E:\GHR\刘金行代码整理\test_data\two_objects_dif\DDL';
    namelist = dir(fullfile(root_dir, '*.csv'));

    len = length(namelist);
    if len == 0
        warning('没在 %s 下找到任何 ddl csv 文件', root_dir);
        gray = [];
        return;
    end

    % 输出目录（图片+Gray）
    path_dir = '../images/two_objects_dif/VGG16/Best_dice_20';
    % path_dir = '../images/two_objects_dif/ECA_ResNet101/Best_dice_30';
    % 原图
    % path_dir = '../images/two_objects_dif/real';

    % 图片目录
    path_image = fullfile(path_dir, 're_image');
    if ~isfolder(path_image)
        mkdir(path_image);
    end

    % Gray 数值目录
    path_gray = fullfile(path_dir, 'Gray');
    if ~isfolder(path_gray)
        mkdir(path_gray);
    end

    % 不弹出绘图窗口
    set(0, 'DefaultFigureVisible', 'off');

    %% ==================== 3. 循环处理每一个 ddl 文件 ====================
    for k = 1:len
        filename = namelist(k).name;
        fprintf('Processing %d / %d: %s\n', k, len, filename);

        % 完整路径
        fullpath = fullfile(namelist(k).folder, filename);

        % 读取 ddl（假设是每个单元一个电导率/电阻率）
        sig = csvread(fullpath);   % 也可以改用 readmatrix(fullpath);
        sig = sig(:);              % 拉成列向量，保证和 elem_data 维度兼容

        % 以 GN 反演结果 img_gn 为模板，只替换 elem_data
        img = img_gn;
        img.elem_data = sig;

        % 画图并保存
        figure;
        a = show_slices(img);

        % 去掉后缀，得到纯文件名
        [name_no_ext, ~] = strtok(filename, '.');

        % 保存图片
        out_png = fullfile(path_image, [name_no_ext, '.png']);
        print_convert(out_png, '-density 60');

        % 保存矩阵/灰度值
        out_mat = fullfile(path_gray, [name_no_ext, '.csv']);
        writematrix(a, out_mat);
    end

    % 返回最后一张图的矩阵
    gray = a;
end
