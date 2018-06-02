%% WRITTEN BY:
%  Nicklas Hansen

function [index, amp] = peak_detect(folder, filename, FS)
    resources = strcat(folder, 'Matlab\Resources');
    edfs = strcat(folder, 'Files\Data\shhs2\polysomnography\edfs');
    addpath(genpath(resources));
    addpath(genpath(edfs));

    [hea,x] = edfread(filename);
    IDX = cellfun(@(a) ~isempty(a), strfind(hea.label,'ECG'));
    tm = RRTachogram.wrapper(x(IDX,:), FS);    
    index = tm.getTM('RR')*FS;
    amp = abs(tm.getX('RR'));
end