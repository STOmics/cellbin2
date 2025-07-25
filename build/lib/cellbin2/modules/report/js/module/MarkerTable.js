
const MarkerTable = {
  props: {
    moduleTitle: String,
    msg: Array,
    data: Array,
    prefix: String
  },
  setup(props) {
    const ifShowExplain = ref(false);
    const tableId = ref(props.prefix + '-marker-table');
    const tableContainerId = ref(props.prefix + '-table-container');
    onMounted(() => {
      console.log('Mount MarkerTable: ', props.prefix);
      const clusterNums = props.data[0].length / 2 - 1;

      let clusterHeaderTemplate = '<th colspan="2" class="cluster-header">Marker</th>';
      for (let i = 0; i < clusterNums; i++) {
        clusterHeaderTemplate += `<th colspan="2" class="cluster-header">Cluster${i + 1}</th>`;
      };

      let geneidHeaderTemplate = "<th class='geneid-header'>ID</th> <th class='geneid-header'>Name</th>";
      for (let i = 0; i < clusterNums; i++) {
        geneidHeaderTemplate += `<th class="geneid-header">L2FC</th>
                                 <th class="geneid-header">p-value</th>`;
      };

      $(`#${tableContainerId.value}`).html(`
          <table id="${tableId.value}" cellpadding="0" cellspacing="0" border="0" class="display">
              <thead>
                  <tr> ${clusterHeaderTemplate} </tr>
                  <tr> ${geneidHeaderTemplate}  </tr>
              </thead>
          </table>`
      );

      $(`#${tableId.value}`).dataTable({
        "data": props.data,
        "scrollX": true
      });

      var rows = document.getElementById(`${tableId.value}`).getElementsByTagName("tr");
      for (var i = 2; i < rows.length; i++) {
        var cells = rows[i].getElementsByTagName("td");
        for (var j = 2; j < cells.length; j += 2) {
          var l2fcValue = parseFloat(cells[j].textContent);
          var pvalue = parseFloat(cells[j + 1].textContent);
          if (l2fcValue < 0 || pvalue > 0.1) {
            cells[j].classList.add("gray-text");
            cells[j + 1].classList.add("gray-text");
          };
        };
      };
    });

    return {
      props,
      tableContainerId,
      ifShowExplain
    };
  },
  template: `
  <div class="module-box" style="width: 1200px; ">
    <div class="module-title-box">
      <div class="title-box-left">
        <span class="title-icon"></span>
        <span class="title-label">Top Markers by Cluster</span>
        <el-button v-if="props.msg && props.msg.length" circle type="info" plain size="small" >
          <el-icon @click="ifShowExplain = !ifShowExplain" size="24">
            <zhrAsk />
          </el-icon>
        </el-button>
      </div>
    </div>

    <div class="module-msg" v-if="ifShowExplain">
      <div v-for="item in props.msg">
      {{ item }}
      </div> 
    </div>
    <div class="module-content-box" style="display: flex;justify-content: space-evenly;">
      <div :id="tableContainerId">
    </div>
  </div>
  `
};