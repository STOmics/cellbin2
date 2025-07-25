
const ExpressionRightMsg = {
  props: {
    data: Array
  },
  setup(props) {
    onMounted(() => {
      console.log('Mount: ', props);
    });
    return {
      props,
    };
  },
  template: `
  <div class="expression-right-msg-box"
    style="box-sizing: border-box;width: 396px; margin-bottom: 24px;">
    <template v-for="item in props.data">
      <div class="sub-box">
        <div class="box-header-color"></div>
        <div class="box-main-content">
          <div class='main-content-value'>{{item.value}}</div>
          <div class='main-content-title'>{{item.label}}</div>
        </div>
      </div>
    </template>
  </div>
  `
};