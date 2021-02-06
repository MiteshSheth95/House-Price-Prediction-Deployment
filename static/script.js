// function getPostedbyValue() {
//     var uiposted = document.getElementsByName("uiposted");
//     for (var i in uiposted) {
//       if (uiposted[i].checked) {
//         return parseInt(i) + 1;
//       }
//     }
//     return -1; // Invalid Value
  //}
  function getUnderConstructionvalue() {
    var uiuc = document.getElementsByName("uiuc");
    for (var i in uiuc) {
      if (uiuc[i].checked) {
        return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
  }

  function getReraValue() {
    var uirera = document.getElementsByName("uirera");
    for (var i in uirera) {
      if (uirera[i].checked) {
        return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
  }

  function getBHKValue() {
    // Get no of BHK's
    var uiBHK = document.getElementsByName("uiBHK");
    for (var i in uiBHK) {
      if (uiBHK[i].checked) {
        return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
  }

  
  function getReadyToMovevalue() {
    var uirtm = document.getElementsByName("uirtm");
    for (var i in uirtm) {
      if (uirtm[i].checked) {
        return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
  }
  
  function getResaleValue() {
    var uiresale = document.getElementsByName("uiresale");
    for (var i in uiresale) {
      if (uiresale[i].checked) {
        return parseInt(i) + 1;
      }
    }
    return -1; // Invalid Value
  }
  
  function onClickedEstimatePrice() {
    debugger;
    //var postedby = getPostedbyValue();
    var underConstruction = getUnderConstructionvalue();
    var rera = getReraValue();
    var bhk = getBHKValue();
    var sqft = document.getElementById("uiSqft");
    var rtm = getReadyToMovevalue();
    var resale= getResaleValue();
    var location = document.getElementById("uiLocations");
    var estPrice = document.getElementById("uiEstimatedPrice");
  
    var url = "http://127.0.0.1:5000/predict_house_price";
    //   var url = "/api/predict_home_price"; // only Deployment
  
    $.post(
      url,
      {
        //postedby: postedby,
        underConstruction: underConstruction,
        rera: rera,
        bhk: bhk,
        sqft: parseFloat(sqft.value),
        readytoMove: rtm,
        resale: resale,
        location: location.value,
      },
      function (data, status) {
        estPrice.innerHTML =
          "<h2>" + data.estimated_price.toString() + " Lakh</h2>";
      }
    );
  }
  
