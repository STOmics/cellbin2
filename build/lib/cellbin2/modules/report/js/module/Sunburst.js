
const Sunburst = {
  props: {
    plotData: Array
  },
  setup(props) {
    const ifShowExplain = ref(false);
    const layout = {
      margin: { l: 0, r: 0, b: 0, t: 0 },
      width: 400,
      height: 400
    };

    const showIcon = {
      width: 24,
      height: 24,
      path: d = "M12,0 C18.627417,0 24,7 24,10 C24,13 18.627417,20 12,20 C5.372583,20 0,13 0,10 C0,7 5.372583,0 12,0 Z M12,4 C8.6862915,4 6,6.6862915 6,10 C6,13.3137085 8.6862915,16 12,16 C15.3137085,16 18,13.3137085 18,10 C18,6.6862915 15.3137085,4 12,4 Z M12,6 C12.4097984,6 12.8052036,6.06162492 13.1774539,6.17611322 C12.4831541,6.49010301 12,7.18862816 12,8 C12,9.1045695 12.8954305,10 14,10 C14.8113718,10 15.509897,9.51684593 15.8235627,8.82255053 C15.9383751,9.19479642 16,9.59020157 16,10 C16,12.209139 14.209139,14 12,14 C9.790861,14 8,12.209139 8,10 C8,7.790861 9.790861,6 12,6 Z",
      transform: 'matrix(1 0 0 1 0 1)'
    };

    const hideIcon = {
      width: 24,
      height: 24,
      path: d = "M5.9330127,0 L17.9330127,20.7846097 L17.0669873,21.2846097 L5.0669873,0.5 L5.9330127,0 Z M5.34211146,2.97822979 L7.48583013,6.68968026 C6.560726,7.74535862 6,9.12834945 6,10.6423048 C6,13.9560133 8.6862915,16.6423048 12,16.6423048 C12.398833,16.6423048 12.7885772,16.6033908 13.1656428,16.5291523 L15.2260808,20.0978505 C14.1995713,20.4432355 13.1176284,20.6423048 12,20.6423048 C5.372583,20.6423048 0,13.6423048 0,10.6423048 C0,8.7572541 2.12122489,5.29290723 5.34211146,2.97822979 Z M12,0.642304845 C18.627417,0.642304845 24,7.64230485 24,10.6423048 C24,12.6644403 21.5590376,16.5039231 17.9381468,18.7920743 L15.8768358,15.2217865 C17.1756018,14.1211947 18,12.478024 18,10.6423048 C18,7.32859635 15.3137085,4.64230485 12,4.64230485 C11.2879292,4.64230485 10.6048302,4.76634744 9.97113223,4.9940032 L7.95339461,1.49814302 C9.21765754,0.963648691 10.5800264,0.642304845 12,0.642304845 Z M8.5742433,8.57616266 L12.076529,14.6415874 L12,14.6423048 C9.790861,14.6423048 8,12.8514438 8,10.6423048 C8,9.88626049 8.20975404,9.17920683 8.5742433,8.57616266 Z M15.8235627,9.46485537 C15.9383751,9.83710126 16,10.2325064 16,10.6423048 C16,11.7347028 15.5620974,12.7248231 14.8522555,13.4467024 L13.1120642,10.4348849 C13.3795959,10.5676587 13.6810807,10.6423048 14,10.6423048 C14.8113718,10.6423048 15.509897,10.1591508 15.8235627,9.46485537 Z M12,6.64230485 C12.4097984,6.64230485 12.8052036,6.70392977 13.1774539,6.81841807 C12.5196659,7.11589576 12.0514001,7.75852837 12.0039689,8.51527534 L10.9962248,6.76929597 C11.3169811,6.68640312 11.6533407,6.64230485 12,6.64230485 Z",
      transform: 'matrix(1 0 0 1 0 1)'
    };

    onMounted(() => {
      console.log('Mount: ', props);
      const config = {
        displaylogo: false,
        modeBarButtonsToAdd: [
          {
            name: "show text",
            icon: showIcon,
            label: "eye",
            click: function (gd) {
              var newTextInfo = { textinfo: "label+text" };
              var noTextInfo = { textinfo: "none" };
              var new_bottom = {
                displaylogo: false,
                modeBarButtonsToAdd: [{
                  name: "hidden text",
                  icon: hideIcon,
                  click: function (gd) {
                    var noTextInfo = { textinfo: "none" };
                    Plotly.react('microbiome-sunburst-plot', props.plotData, layout, config);
                    Plotly.restyle(gd, noTextInfo);
                  }
                }]
              };
              Plotly.react('microbiome-sunburst-plot', props.plotData, layout, new_bottom);
              Plotly.restyle(gd, newTextInfo);
            }
          },
        ]
      };
      Plotly.newPlot('microbiome-sunburst-plot', props.plotData, layout, config);
    });

    return {
      props,
      ifShowExplain
    };
  },
  template: `
    <div class="module-content-right" style=" width: 576px;height: calc(100% - 24px);display: flex;justify-content: center;align-items: center;background-color: #ffffff;">
      <div class="microbiome-plot" id="microbiome-sunburst-plot" style="width: 400px;height: 400px;">
      </div>
    </div>
  `
};