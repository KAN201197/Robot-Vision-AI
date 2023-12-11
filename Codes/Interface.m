classdef Interface < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure               matlab.ui.Figure
        RightPanel             matlab.ui.container.Panel
        TextArea_2             matlab.ui.control.TextArea
        AIButtonGroup          matlab.ui.container.ButtonGroup
        %SVMButton              matlab.ui.control.RadioButton
        CNNButton              matlab.ui.control.RadioButton
        NonButton_2            matlab.ui.control.RadioButton
        CloseButton            matlab.ui.control.Button
        ProcessButtonGroup     matlab.ui.container.ButtonGroup
        SegmentButton          matlab.ui.control.RadioButton
        OutlineButton          matlab.ui.control.RadioButton
        NonButton              matlab.ui.control.RadioButton
        ThredholdLabel         matlab.ui.control.Label
        Slider                 matlab.ui.control.Slider
        BinaryCheckBox         matlab.ui.control.CheckBox
        LowerHalfCheckBox      matlab.ui.control.CheckBox
        TextArea               matlab.ui.control.TextArea
        SmoothMaskButtonGroup  matlab.ui.container.ButtonGroup
        Button5                matlab.ui.control.RadioButton
        Button4                matlab.ui.control.RadioButton
        Button3                matlab.ui.control.RadioButton
        Button2                matlab.ui.control.RadioButton
        Label                  matlab.ui.control.Label
        OpenFileButton         matlab.ui.control.Button
        LeftPanel              matlab.ui.container.Panel
        UIAxes_2               matlab.ui.control.UIAxes
        UIAxes                 matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
        img_processing
        maskSize
        img_Original % Description
        Process % Description
        state_ai % Description
        characters = ['H' 'D' '4' '4' '7' '8' '0' 'A' '0' '0'] % Description
        net_cnn % Description
    end
    
    methods (Access = private)
        
        function Main_Processes(app)
            
            app.img_processing = app.img_Original;
            num_clusters=10;
            if app.LowerHalfCheckBox.Value == 0
                app.img_processing = app.img_processing;
            else
                imheight = size(app.img_processing,1);
                app.img_processing = app.img_processing(round(imheight/2):imheight,:);
            end
%            app.Label.Text = app.maskSize;
            if app.maskSize ~= 1
                % Create the mask
                mask = ones(app.maskSize) / app.maskSize^2;
                
                % Define the angles to test
                angles = [0, 45, 90, 135, 180, 225, 270, 315, -1];
                
                % Initialize the minimum dispersion and best angle
                minDispersion = Inf;
                bestAngle = 0;
                
                % Loop through all angles
                for i = 1:length(angles)
                    
                    % Get the angle to test
                    angle = angles(i);
                    
                    % If the angle is -1, use the unrotated mask
                    if angle == -1
                        rotatedMask = mask;
                    else
                        % Otherwise, rotate the mask
                        rotatedMask = imrotate(mask, angle, 'crop');
                    end
                
                    % Apply the filter to the image
                    filteredImg = convn(double(app.img_processing), rotatedMask, 'same');
                    
                    % Calculate the dispersion of the filtered image
                    dispersion = std2(filteredImg);
                    
                    % Update the minimum dispersion and best angle if necessary
                    if dispersion < minDispersion
                        minDispersion = dispersion;
                        bestAngle = angle;
                    end
                end
                
                % Apply the filter to the image with the best angle
                if bestAngle == -1
                    filteredImg = convn(double(app.img_processing), mask, 'same');
                    str = {'Average mask has the minimum dispersion. \n';['Dispersion =',minDispersion]};
                    app.TextArea.Value = str;
                else
                    rotatedMask = imrotate(mask, bestAngle, 'crop');
                    filteredImg = convn(double(app.img_processing), rotatedMask, 'same');
                    str = {['Rotating mask at ',num2str(bestAngle),'° has the minimum dispersion.'];['Dispersion =',num2str(minDispersion)]};
                    app.TextArea.Value = str;
                end
                
                app.img_processing = filteredImg;
            else
                app.TextArea.Value = '';
            end    
            
            if app.BinaryCheckBox.Value ~= 0
                threshold = app.Slider.Value;
                app.ThredholdLabel.Text = ['Thredhold = ',num2str(threshold)];
                app.img_processing = imbinarize(app.img_processing,threshold);
            end

            if app.Process == 1
                app.img_processing = edge(app.img_processing,"log");
                
            elseif app.Process == 2
                delete('files/segmented_chars/*.png')
                imageSeperated = app.img_processing;
                CC = bwconncomp(app.img_processing);
                raws = size(app.img_processing,1);
                number = CC.NumObjects;
                for i=1:number
                    cluster = cell2mat(CC.PixelIdxList(i));
                    startPoint = cluster(1);
                    endPoint = cluster(end);
                    startColumn = round(startPoint/raws)-1;
                    endColumn = round(endPoint/raws)-1;
                    clusterWidth = endColumn-startColumn;
                    %middleColumn = startColumn+round(clusterWidth/2);
                    if clusterWidth>240
                        cutColumns = 100;
                        middleColumns = startColumn + cutColumns : cutColumns : startColumn + clusterWidth - cutColumns;
                        for cut_num = 1 : length(middleColumns)-1
                            middleColumn1 = middleColumns(cut_num);
                            middleColumn2 = middleColumns(cut_num+1);
                            imageSeparated(:,middleColumn1:middleColumn1+4) = 0;
                            imageSeparated(:,middleColumn2-4:middleColumn2) = 0;
                        end
                    elseif (clusterWidth<=240) && (clusterWidth>130)
                        middleColumn = startColumn+round(clusterWidth/2);
                        imageSeperated(:,middleColumn:middleColumn+4) = 0;
                    
                    elseif clusterWidth<30
                        imageSeperated(CC.PixelIdxList{i}) = 0;
                    end
                end
                CC = bwconncomp(imageSeperated);
                % loop through each connected component and save it as a separate image
                images = cell(5);
                num_clusters = CC.NumObjects;
                for i=1:num_clusters

                    % create a binary image containing only the i-th connected component
                    binaryImage = false(size(imageSeperated));
                    binaryImage(CC.PixelIdxList{i}) = true;
                    % use the connected component bounding box to crop the image
                    stats = regionprops(binaryImage, 'BoundingBox');
                    bbox = stats.BoundingBox;
                    croppedImage = imcrop(binaryImage, bbox);
                    % pad the cropped image with black border to make it 128x128
                    [height, width] = size(croppedImage);
                    if height > width
                        border = floor((height-width)/2);
                        resizedImage = imresize(croppedImage, [128 NaN]);
                        paddedImage = padarray(resizedImage, [0 border], 0, 'both');
                        paddedImage = padarray(paddedImage, [0 128-min(size(paddedImage,2),128)], 0, 'post');
                    else
                        border = floor((width-height)/2);
                        resizedImage = imresize(croppedImage, [NaN 128]);
                        paddedImage = padarray(resizedImage, [border 0], 0, 'both');
                        paddedImage = padarray(paddedImage, [128-min(size(paddedImage,1),128) 0], 0, 'post');
                    end
                    images{i} = paddedImage;
                    % save the padded image as a separate file in the new directory
                    imwrite(paddedImage, fullfile('files/segmented_chars', ['char_', num2str(i), '.png']));
                    
                end
                L = labelmatrix(CC);
                app.img_processing = label2rgb(L);
            end
            
            imshow(app.img_processing,"Parent",app.UIAxes_2)

            if  num_clusters~=10
                app.TextArea_2.Value = 'Segment Error!';
            elseif app.state_ai == 0
                app.TextArea_2.Value = '';
            elseif app.state_ai == 1
                num_correct = 0;
                results_identified = zeros(1,10);
                for i = 1:10
                    % Load a new image to classify
                    newImgGray = images{i};
                    newImgGray = uint8(newImgGray) * 255;
                    newImgGray = imresize(newImgGray, [128 128]);
                    % Classify the new image using the trained CNN
                    predictedLabel = classify(app.net_cnn, newImgGray);
                    results_identified(i) = char(predictedLabel);
                    if predictedLabel == app.characters(i)
                        num_correct = num_correct+1;
                    end
                end
                results_str = results_identified;
                accuracy_str = num2str(num_correct*10);
                app.TextArea_2.Value = ['Results: ' results_str newline 'Accuracy: ' accuracy_str '%'];
            %elseif app.state_ai == 2

            end
            app.AIButtonGroup.SelectedObject = app.NonButton_2;
            app.state_ai = 0;

            
    

        end
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            load('files\CNN.mat', 'net');
            app.net_cnn = net;

        end

        % Button pushed function: OpenFileButton
        function OpenFileButtonPushed(app, event)
            [filename,pathname] = uigetfile(["*.bmp";"*.jpg";"*.jpeg";"*.png"],"Choose an image");
            inputim = imread(fullfile(pathname,filename));
            inputim = im2double(rgb2gray(inputim));
            app.img_processing = inputim;
            app.img_Original = inputim;
            imshow(inputim,"Parent",app.UIAxes)
            app.Label.Text = filename;
            app.LowerHalfCheckBox.Enable = "on";
            app.SmoothMaskButtonGroup.Enable = "on";
            app.BinaryCheckBox.Enable = "on";
            app.ProcessButtonGroup.Enable = "on";
        end

        % Value changed function: LowerHalfCheckBox
        function LowerHalfCheckBoxValueChanged(app, event)
            %value = app.LowerHalfCheckBox.Value;
            Main_Processes(app)
        end

        % Button pushed function: CloseButton
        function CloseButtonPushed(app, event)
            delete(app.UIFigure)
            
        end

        % Selection changed function: SmoothMaskButtonGroup
        function SmoothMaskButtonGroupSelectionChanged(app, event)
            selectedButton = app.SmoothMaskButtonGroup.SelectedObject;
            switch selectedButton.Text
                case "3x3", app.maskSize = 3;
                case "7x7", app.maskSize = 7;
                case "11x11", app.maskSize = 11;
                otherwise, app.maskSize = 1;
            end
            Main_Processes(app)
        end

        % Value changed function: BinaryCheckBox
        function BinaryCheckBoxValueChanged(app, event)
            value = app.BinaryCheckBox.Value;
            if value ~= 0
                app.Slider.Enable = "on";
            else
                app.Slider.Enable = "off";
            end
            Main_Processes(app)
        end

        % Callback function: Slider, Slider
        function SliderValueChanging(app, event)
            %changingValue = event.Value;
%             app.AIButtonGroup.SelectedObject = app.NonButton_2;
            %app.NonButton_2.Value = true;
            Main_Processes(app)
        end

        % Selection changed function: ProcessButtonGroup
        function ProcessButtonGroupSelectionChanged(app, event)
            selectedButton = app.ProcessButtonGroup.SelectedObject;
            switch selectedButton.Text
                case "Non", app.Process = 0; app.AIButtonGroup.Enable = "off";
                case "Outline", app.Process = 1; app.AIButtonGroup.Enable = "off";
                case "Segment", app.Process = 2; app.AIButtonGroup.Enable = "on";
            end
            Main_Processes(app)
        end

        % Selection changed function: AIButtonGroup
        function AIButtonGroupSelectionChanged(app, event)
            selectedButton = app.AIButtonGroup.SelectedObject;
            switch selectedButton.Text
                case "Non", app.state_ai = 0;
                case "CNN", app.state_ai = 1;
                %case "SVM", app.state_ai = 2;
            end
            Main_Processes(app)
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.AutoResizeChildren = 'off';
            app.UIFigure.Color = [1 1 1];
            app.UIFigure.Position = [100 100 1300 600];
            app.UIFigure.Name = 'MATLAB App';
            app.UIFigure.Resize = 'off';

            % Create LeftPanel
            app.LeftPanel = uipanel(app.UIFigure);
            app.LeftPanel.AutoResizeChildren = 'off';
            app.LeftPanel.Position = [1 1 970 600];

            % Create UIAxes
            app.UIAxes = uiaxes(app.LeftPanel);
            app.UIAxes.MinorGridLineStyle = 'none';
            app.UIAxes.XAxisLocation = 'origin';
            app.UIAxes.XColor = 'none';
            app.UIAxes.YAxisLocation = 'origin';
            app.UIAxes.YColor = 'none';
            app.UIAxes.ZColor = 'none';
            app.UIAxes.ClippingStyle = 'rectangle';
            app.UIAxes.TickDir = 'none';
            app.UIAxes.GridColor = 'none';
            app.UIAxes.MinorGridColor = 'none';
            app.UIAxes.Visible = 'off';
            app.UIAxes.Position = [1 300 968 299];

            % Create UIAxes_2
            app.UIAxes_2 = uiaxes(app.LeftPanel);
            app.UIAxes_2.MinorGridLineStyle = 'none';
            app.UIAxes_2.XAxisLocation = 'origin';
            app.UIAxes_2.XColor = 'none';
            app.UIAxes_2.YAxisLocation = 'origin';
            app.UIAxes_2.YColor = 'none';
            app.UIAxes_2.ZColor = 'none';
            app.UIAxes_2.ClippingStyle = 'rectangle';
            app.UIAxes_2.TickDir = 'none';
            app.UIAxes_2.GridColor = 'none';
            app.UIAxes_2.MinorGridColor = 'none';
            app.UIAxes_2.Visible = 'off';
            app.UIAxes_2.Position = [1 0 968 299];

            % Create RightPanel
            app.RightPanel = uipanel(app.UIFigure);
            app.RightPanel.AutoResizeChildren = 'off';
            app.RightPanel.Position = [969 1 332 600];

            % Create OpenFileButton
            app.OpenFileButton = uibutton(app.RightPanel, 'push');
            app.OpenFileButton.ButtonPushedFcn = createCallbackFcn(app, @OpenFileButtonPushed, true);
            app.OpenFileButton.FontSize = 14;
            app.OpenFileButton.FontWeight = 'bold';
            app.OpenFileButton.Position = [10 547 78 25];
            app.OpenFileButton.Text = 'Open File';

            % Create Label
            app.Label = uilabel(app.RightPanel);
            app.Label.FontSize = 14;
            app.Label.Position = [95 549 126 22];
            app.Label.Text = '';

            % Create SmoothMaskButtonGroup
            app.SmoothMaskButtonGroup = uibuttongroup(app.RightPanel);
            app.SmoothMaskButtonGroup.AutoResizeChildren = 'off';
            app.SmoothMaskButtonGroup.SelectionChangedFcn = createCallbackFcn(app, @SmoothMaskButtonGroupSelectionChanged, true);
            app.SmoothMaskButtonGroup.Enable = 'off';
            app.SmoothMaskButtonGroup.Title = 'Smooth Mask';
            app.SmoothMaskButtonGroup.FontWeight = 'bold';
            app.SmoothMaskButtonGroup.FontSize = 14;
            app.SmoothMaskButtonGroup.Position = [8 464 316 62];

            % Create Button2
            app.Button2 = uiradiobutton(app.SmoothMaskButtonGroup);
            app.Button2.Text = 'Non';
            app.Button2.FontSize = 14;
            app.Button2.Position = [10 14 48 22];
            app.Button2.Value = true;

            % Create Button3
            app.Button3 = uiradiobutton(app.SmoothMaskButtonGroup);
            app.Button3.Text = '3x3';
            app.Button3.FontSize = 14;
            app.Button3.Position = [87 14 44 22];

            % Create Button4
            app.Button4 = uiradiobutton(app.SmoothMaskButtonGroup);
            app.Button4.Text = '7x7';
            app.Button4.FontSize = 14;
            app.Button4.Position = [160 14 44 22];

            % Create Button5
            app.Button5 = uiradiobutton(app.SmoothMaskButtonGroup);
            app.Button5.Text = '11x11';
            app.Button5.FontSize = 14;
            app.Button5.Position = [232 14 58 22];

            % Create TextArea
            app.TextArea = uitextarea(app.RightPanel);
            app.TextArea.Editable = 'off';
            app.TextArea.FontSize = 14;
            app.TextArea.Position = [8 395 316 60];

            % Create LowerHalfCheckBox
            app.LowerHalfCheckBox = uicheckbox(app.RightPanel);
            app.LowerHalfCheckBox.ValueChangedFcn = createCallbackFcn(app, @LowerHalfCheckBoxValueChanged, true);
            app.LowerHalfCheckBox.Enable = 'off';
            app.LowerHalfCheckBox.Text = 'Lower Half';
            app.LowerHalfCheckBox.FontSize = 14;
            app.LowerHalfCheckBox.FontWeight = 'bold';
            app.LowerHalfCheckBox.Position = [232 549 94 22];

            % Create BinaryCheckBox
            app.BinaryCheckBox = uicheckbox(app.RightPanel);
            app.BinaryCheckBox.ValueChangedFcn = createCallbackFcn(app, @BinaryCheckBoxValueChanged, true);
            app.BinaryCheckBox.Enable = 'off';
            app.BinaryCheckBox.Text = 'Binary';
            app.BinaryCheckBox.FontSize = 14;
            app.BinaryCheckBox.FontWeight = 'bold';
            app.BinaryCheckBox.Position = [8 349 66 22];

            % Create Slider
            app.Slider = uislider(app.RightPanel);
            app.Slider.Limits = [0 1];
            app.Slider.ValueChangedFcn = createCallbackFcn(app, @SliderValueChanging, true);
            app.Slider.ValueChangingFcn = createCallbackFcn(app, @SliderValueChanging, true);
            app.Slider.Enable = 'off';
            app.Slider.Position = [10 327 312 3];
            app.Slider.Value = 0.35;

            % Create ThredholdLabel
            app.ThredholdLabel = uilabel(app.RightPanel);
            app.ThredholdLabel.FontSize = 14;
            app.ThredholdLabel.Position = [125 349 171 22];
            app.ThredholdLabel.Text = 'Thredhold';

            % Create ProcessButtonGroup
            app.ProcessButtonGroup = uibuttongroup(app.RightPanel);
            app.ProcessButtonGroup.AutoResizeChildren = 'off';
            app.ProcessButtonGroup.SelectionChangedFcn = createCallbackFcn(app, @ProcessButtonGroupSelectionChanged, true);
            app.ProcessButtonGroup.Enable = 'off';
            app.ProcessButtonGroup.Title = 'Process';
            app.ProcessButtonGroup.FontWeight = 'bold';
            app.ProcessButtonGroup.FontSize = 14;
            app.ProcessButtonGroup.Position = [9 219 315 59];

            % Create NonButton
            app.NonButton = uiradiobutton(app.ProcessButtonGroup);
            app.NonButton.Text = 'Non';
            app.NonButton.FontSize = 14;
            app.NonButton.Position = [11 11 58 22];
            app.NonButton.Value = true;

            % Create OutlineButton
            app.OutlineButton = uiradiobutton(app.ProcessButtonGroup);
            app.OutlineButton.Text = 'Outline';
            app.OutlineButton.FontSize = 14;
            app.OutlineButton.Position = [106 10 66 22];

            % Create SegmentButton
            app.SegmentButton = uiradiobutton(app.ProcessButtonGroup);
            app.SegmentButton.Text = 'Segment';
            app.SegmentButton.FontSize = 14;
            app.SegmentButton.Position = [206 10 78 22];

            % Create CloseButton
            app.CloseButton = uibutton(app.RightPanel, 'push');
            app.CloseButton.ButtonPushedFcn = createCallbackFcn(app, @CloseButtonPushed, true);
            app.CloseButton.FontSize = 18;
            app.CloseButton.FontWeight = 'bold';
            app.CloseButton.FontColor = [1 0 0];
            app.CloseButton.Position = [225 17 88 53];
            app.CloseButton.Text = 'Close';

            % Create AIButtonGroup
            app.AIButtonGroup = uibuttongroup(app.RightPanel);
            app.AIButtonGroup.AutoResizeChildren = 'off';
            app.AIButtonGroup.SelectionChangedFcn = createCallbackFcn(app, @AIButtonGroupSelectionChanged, true);
            app.AIButtonGroup.Enable = 'off';
            app.AIButtonGroup.Title = 'AI';
            app.AIButtonGroup.FontWeight = 'bold';
            app.AIButtonGroup.FontSize = 14;
            app.AIButtonGroup.Position = [9 140 315 59];

            % Create NonButton_2
            app.NonButton_2 = uiradiobutton(app.AIButtonGroup);
            app.NonButton_2.Text = 'Non';
            app.NonButton_2.FontSize = 14;
            app.NonButton_2.Position = [11 9 58 22];
            app.NonButton_2.Value = true;

            % Create CNNButton
            app.CNNButton = uiradiobutton(app.AIButtonGroup);
            app.CNNButton.Text = 'CNN';
            app.CNNButton.FontSize = 14;
            app.CNNButton.Position = [106 8 52 22];

%             % Create SVMButton
%             app.SVMButton = uiradiobutton(app.AIButtonGroup);
%             app.SVMButton.Text = 'SVM';
%             app.SVMButton.FontSize = 14;
%             app.SVMButton.Position = [206 8 52 22];

            % Create TextArea_2
            app.TextArea_2 = uitextarea(app.RightPanel);
            app.TextArea_2.Editable = 'off';
            app.TextArea_2.FontSize = 14;
            app.TextArea_2.Position = [9 88 316 42];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = Interface

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end