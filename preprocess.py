import numpy as np
import tensorflow as tf
class MotorImageryDataset:
    def __init__(self,paths):
        self.paths = paths
        self.X = []
        self.y = np.array([])
                
        
    def create(self,window=False):
        event_codes={'left':769,'right':770,'foot':771,'tongue':772}
        for path in self.paths:
            dataset = np.load(path)
            for event in event_codes.values():
                event_signals_3d = self.get_events_signals(dataset,event)
                event_targets = [event for i in range(len(event_signals_3d))]
                self.X.extend(event_signals_3d)
                self.y = np.append(self.y,event_targets)
                
        self.X = np.array(self.X) 
        
        
        if window:
            input_data = np.array([self.create_window(signal) for signal in self.X])
            labels = [[l]*14 for l in self.y]
        else:
            input_data = np.array(self.X) 
            labels = [[l] for l in self.y]
        
        labels = np.array(labels).reshape(-1,1).reshape(-1,1)
        labels = tf.one_hot(labels-769, depth=4)
        labels = tf.reshape(labels,shape=[len(labels),4]).numpy()
        

        with open('integrated_data.npy', 'wb') as f:
            np.save(f, input_data)
            np.save(f, labels)
        
    def create_window(self,signal_matrix,length=300):
    
        window =[]
        for i in range (len(signal_matrix[0]) - length + 1):
            window.append(signal_matrix[:,i:length+i].T)

        window = np.array(window)
        return window
    
    
    def get_events_signals(self,data,event_type,n_channels=22):
        selected_events_boolean = data['etyp'] == event_type
        selected_events_boolean = selected_events_boolean.reshape(len(selected_events_boolean),1)
        indexes = [i for i, x in enumerate(selected_events_boolean) if x]
        positions = data['epos'].reshape(len(data['epos']))
        durations = data['edur'].reshape(len(data['edur']))
        event_signals_3d =np.array([data['s'].T[:n_channels,positions[i]:positions[i]+durations[i] ] for i in indexes])
        return event_signals_3d