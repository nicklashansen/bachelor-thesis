%% WRITTEN BY:
%  Nicklas Hansen

function [index, amp] = peak_detect(folder, filename, FS)
    resources = strcat(folder, 'Matlab\Resources');
    edfs = strcat(folder, 'Files\Data\mesa\polysomnography\edfs');
    addpath(genpath(resources));
    addpath(genpath(edfs));

    [hea,x] = edfread(filename);
    IDX = cellfun(@(a) ~isempty(a), strfind(hea.label,'EKG'));
    tm = RRTachogram.wrapper(x(1,:), FS);    
    index = tm.getTM('RR')*FS;
    amp = abs(tm.getX('RR'));
end
